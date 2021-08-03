vector prepare_unit_log_vector(vector y){
  vector[rows(y)+2] rv;
  rv[1] = y[1];
  rv[2:5] = append_row(y[2:4], -sum(y[2:4]));
  rv[6:7] = [y[5], -y[5]]';
  rv[8:] = y[6:];
  return rv;
}
vector constrain_log_vector(vector y){
  vector[rows(y)] rv = y;
  rv[2:5] = log_softmax(rv[2:5]);
  rv[6:7] = log_softmax(rv[6:7]);
  return rv;
}
vector dydt_exposure(
  real t, vector concentration_out,
  vector FVP, vector FFPF, real CFFPF,
  real VMI, real KMI
){
  return FVP .* (
      dot_product(FFPF, concentration_out) + CFFPF - concentration_out
  ) + [
      0,0,0,VMI * concentration_out[4] / (KMI + concentration_out[4])
  ]';
}
vector dydt_nonexposure(
  real t, vector concentration_out,
  vector FVP, vector FFPF,
  real VMI, real KMI
){
  return FVP .* (
      dot_product(FFPF, concentration_out) - concentration_out
  ) + [
      0,0,0,VMI * concentration_out[4] / (KMI + concentration_out[4])
  ]';
}
real lambert_w0_exp(real earg){
  if(is_nan(earg)){return earg;}
  // https://www.wolframalpha.com/input/?i=N%5BTable%5B%7BProductLog%5BExp%5Bx%5D%5D%2CProductLog%5BExp%5Bx%5D%5D%2F%28ProductLog%5BExp%5Bx%5D%5D%2B1%29%7D%2C+%7Bx%2C%7B-40%2C700%7D%7D%5D%2C+32%5D
  // (4.2483542552915889772807209044045×10^-18 | 4.2483542552915889592322070259504×10^-18
  // 693.45830887902549833674969928122 | 0.99856002874871758528483054395180)
  if(earg > 700){
    real x0 = 700;
    real y0 = lambert_w0(exp(x0));
    real dydx = y0/(y0+1);
    return y0 + (earg - x0) * dydx;
    // real y0 = 693.45830887902549833674969928122;
    // real dydx = 0.9985600287487175852848305439518009385629359235843258625030446976;
    // // https://www.wolframalpha.com/input/?i=ProductLog%5BExp%5B700%5D%5D%2F%28ProductLog%5BExp%5B700%5D%5D%2B1%29
    // return y0 + (earg - x0) * dydx;
  }
  if(earg < -40){
    real x0 = -40;
    real y0 = lambert_w0(exp(x0));
    // real y0 = 4.2483542552915889772807209044045064097730030864961656941893e-18;
    return y0 * exp(earg - x0);
  }
  return lambert_w0(exp(earg));
}
real min_concentration(){return 1e-12;}
real exact_michaelis_menten_solution(real dt, real C, real V, real K){
  if(C <= min_concentration()){return min_concentration();}
  if(K == 0){return C - dt * V;}
  real earg = (dt*V+C)/K+log(C/K);
  return K * lambert_w0_exp(earg);
}
vector ginterpolate(real xi, vector left, vector right){
  return exp((1-xi)*log(min_concentration()+left)+xi*log(min_concentration()+right));
}
// vector strang_step(real dt)
real[,] simulate_person(
    data real concentration_exposure,
    data real[] times,
    vector params,
    data vector measured_params,
    data int no_sub_steps
){
  int no_times = size(times);
  int no_exposure_times = 1;
  real rv[no_times, 2];
  real lean_body_mass = measured_params[1]; #[kg]
  real mass_fraction_fat = measured_params[2]; #[1]
  real volume_flow_pulmonary = measured_params[3]; #[l/min]
  // real concentration_exposure = measured_params[4]; #[mug/l]

  real body_mass = lean_body_mass / (1 - mass_fraction_fat); #[kg]
  real volume_fat = mass_fraction_fat * body_mass / .92; #[l]
  real volume_flow_alveolar = .7 * volume_flow_pulmonary; #[l/min]
  real mass_flow_exposure = volume_flow_alveolar * concentration_exposure; #[mug/min]

  real VPR = params[1]; #[1]
  real volume_flow_venous = volume_flow_alveolar / VPR; #[l/min]
  vector[4] unit_volume_flow = params[2:5];
  vector[4] volume_flow = unit_volume_flow * volume_flow_venous; #[l/min]
  # body density = water density = 1kg/l
  vector[3] dummy = lean_body_mass * params[6:8]; #[l]
  vector[4] volume = append_row( #[l]
    dummy[1:2],
    [volume_fat, dummy[3]]'
  );
  real partition_coefficient_alveolar = params[9]; #[1]
  vector[4] partition_coefficient = params[10:13]; #[1]
  vector[4] effective_volume = volume .* partition_coefficient;
  real VMI = pow(lean_body_mass, .7) * params[14]/(effective_volume[4]);
  real KMI = params[15]/effective_volume[4];

  vector[4] concentration_out = rep_vector(min_concentration(), 4); #[mug/l]
  vector[4] FVP = volume_flow ./ (effective_volume); #[1/min]
  real FPF = volume_flow_venous + volume_flow_alveolar / partition_coefficient_alveolar; #[l/min]

  array[no_times] vector[4] all_concentration_out;
  if(no_sub_steps <= 0){
    all_concentration_out[1] = ode_bdf_tol(
      dydt_exposure, concentration_out, 0, {times[1]},
      pow(10, no_sub_steps), pow(10, no_sub_steps-14), 100000000,
      FVP, volume_flow / FPF, mass_flow_exposure / FPF,
      -VMI, KMI
    )[1];
    all_concentration_out[2:] = ode_bdf_tol(
      dydt_nonexposure, all_concentration_out[1], 0, to_array_1d(
        to_vector(times[2:])-times[1]
      ),
      pow(10, no_sub_steps), pow(10, no_sub_steps-14), 100000000,
      FVP, volume_flow / FPF,
      -VMI, KMI
    );
  }else{
    matrix[4,4] A = add_diag( #[1/min]
      FVP * (volume_flow / FPF)',
      -FVP
    );
    vector[4] A_source = mass_flow_exposure / FPF * A \ FVP; #[mug/l]
    vector[4] last_concentration_out = concentration_out; #[mug/l]
    real dt = times[1] / no_sub_steps; #[min]
    matrix[4,4] transition_matrix = matrix_exp(dt * A); #[1]
    vector[4] exp_A_source = transition_matrix * A_source - A_source; #[mug/l]
    real last_time = 0;  #[min]
    real next_time = 0;
    int time_idx = 1;
    real next_checkpoint = times[time_idx];
    while(time_idx <= no_times){
      next_time = last_time + dt;
      concentration_out[4] = exact_michaelis_menten_solution(
        dt/2, concentration_out[4], -VMI, KMI
      );
      if(time_idx <= no_exposure_times){
        concentration_out = transition_matrix * concentration_out + exp_A_source;
      }else{
        concentration_out = transition_matrix * concentration_out;
      }
      concentration_out[4] = exact_michaelis_menten_solution(
        dt/2, concentration_out[4], -VMI, KMI
      );
      while(next_time >= next_checkpoint){
        all_concentration_out[time_idx] = ginterpolate(
          (next_checkpoint - last_time)/dt,
          last_concentration_out,
          concentration_out
        );
        if(time_idx == no_exposure_times){
          concentration_out = all_concentration_out[time_idx];
          next_time = times[time_idx];
        }
        time_idx += 1;
        if(time_idx <= no_times){
          next_checkpoint = times[time_idx];
        }else{
          break;
        }
      }
      last_time = next_time;
      last_concentration_out = concentration_out;
    }
  }
  for(time_idx in 1:no_times){
    concentration_out = all_concentration_out[time_idx];
    real concentration_venous = dot_product(unit_volume_flow, concentration_out);
    real concentration_inhale = (time_idx <= no_exposure_times ? concentration_exposure : 0);
    real concentration_alveolar = (
      concentration_inhale + concentration_venous/VPR
    ) / (
      partition_coefficient_alveolar/VPR + 1
    );
    real concentration_exhale = .7 * concentration_alveolar + .3 * concentration_inhale;
    rv[time_idx] = {min_concentration()+concentration_venous, min_concentration()+concentration_exhale};
  }
  return rv;
}

vector[] new_simulate_person(
    data real exposure,
    data real exposure_time,
    data vector times,
    vector params,
    data vector measured_params,
    data real dt,
    data real alveolar_weight
){
  int no_times = size(times);
  array[2] vector[no_times] rv;
  real lean_body_mass = measured_params[1]; #[kg]
  real mass_fraction_fat = measured_params[2]; #[1]
  real volume_flow_pulmonary = measured_params[3]; #[l/min]
  // real concentration_exposure = measured_params[4]; #[mug/l]

  real body_mass = lean_body_mass / (1 - mass_fraction_fat); #[kg]
  real volume_fat = mass_fraction_fat * body_mass / .92; #[l]
  real volume_flow_alveolar = .7 * volume_flow_pulmonary; #[l/min]
  real mass_flow_exposure = volume_flow_alveolar * exposure; #[mug/min]

  real VPR = params[1]; #[1]
  real volume_flow_venous = volume_flow_alveolar / VPR; #[l/min]
  vector[4] unit_volume_flow = params[2:5];
  vector[4] volume_flow = unit_volume_flow * volume_flow_venous; #[l/min]
  # body density = water density = 1kg/l
  vector[3] dummy = lean_body_mass * params[6:8]; #[l]
  vector[4] volume = append_row( #[l]
    dummy[1:2],
    [volume_fat, dummy[3]]'
  );
  real partition_coefficient_alveolar = params[9]; #[1]
  vector[4] partition_coefficient = params[10:13]; #[1]
  vector[4] effective_volume = volume .* partition_coefficient;
  real VMI = pow(lean_body_mass, .7) * params[14]/(effective_volume[4]);
  real KMI = params[15]/effective_volume[4];

  vector[4] concentration_out = rep_vector(min_concentration(), 4); #[mug/l]
  vector[4] FVP = volume_flow ./ (effective_volume); #[1/min]
  real FPF = volume_flow_venous + volume_flow_alveolar / partition_coefficient_alveolar; #[l/min]

  array[no_times] vector[4] all_concentration_out;
  matrix[4,4] A = add_diag( #[1/min]
    FVP * (volume_flow / FPF)',
    -FVP
  );
  vector[4] A_source = mass_flow_exposure / FPF * A \ FVP; #[mug/l]
  vector[4] last_concentration_out = concentration_out; #[mug/l]
  matrix[4,4] transition_matrix = matrix_exp(dt * A); #[1]
  vector[4] exp_A_source = transition_matrix * A_source - A_source; #[mug/l]
  real last_time = 0;  #[min]
  real next_time = 0;
  {
  int time_idx = 1;
  real next_checkpoint = times[time_idx];
  while(time_idx <= no_times){
    next_time = last_time + dt;
    concentration_out[4] = exact_michaelis_menten_solution(
      dt/2, concentration_out[4], -VMI, KMI
    );
    if(last_time < exposure_time){
      concentration_out = transition_matrix * concentration_out + exp_A_source;
    }else{
      concentration_out = transition_matrix * concentration_out;
    }
    concentration_out[4] = exact_michaelis_menten_solution(
      dt/2, concentration_out[4], -VMI, KMI
    );
    while(next_time >= next_checkpoint){
      all_concentration_out[time_idx] = ginterpolate(
        (next_checkpoint - last_time)/dt,
        last_concentration_out,
        concentration_out
      );
      if(last_time < exposure_time && exposure_time <= next_time){
        concentration_out = all_concentration_out[time_idx];
        next_time = times[time_idx];
      }
      time_idx += 1;
      if(time_idx <= no_times){
        next_checkpoint = times[time_idx];
      }else{
        break;
      }
    }
    last_time = next_time;
    last_concentration_out = concentration_out;
  }
  }
  for(time_idx in 1:no_times){
    concentration_out = all_concentration_out[time_idx];
    real concentration_venous = dot_product(unit_volume_flow, concentration_out);
    real concentration_inhale = (times[time_idx] <= exposure_time ? exposure : 0);
    real concentration_alveolar = (
      concentration_inhale + concentration_venous/VPR
    ) / (
      partition_coefficient_alveolar/VPR + 1
    );
    real concentration_exhale = alveolar_weight * concentration_alveolar + (1-alveolar_weight) * concentration_inhale;
    rv[, time_idx] = {min_concentration()+concentration_venous, min_concentration()+concentration_exhale};
  }
  return rv;
}
