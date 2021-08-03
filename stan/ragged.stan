functions {
  #include basis.stan
}
data {
  int no_persons;
  int no_measured_params;
  array[no_persons] vector[no_measured_params] measured_params;
  int no_experiments;
  int total_no_measurements;
  int person_idxs[no_experiments];
  int measurement_idxs[no_experiments];
  real alveolar_weights[no_experiments];
  int no_measurements[no_experiments];
  vector[no_experiments] exposures;
  vector[no_experiments] exposure_times;
  vector[total_no_measurements] measurement_times;
  array[2] vector[total_no_measurements] measurements;

  int no_latent_params;
  vector<lower=0>[no_latent_params] population_eM_eM;
  vector<lower=0>[no_latent_params] population_eM_eS;
  vector<lower=1>[no_latent_params] population_eS_mu;
  vector<lower=0>[no_latent_params] population_eS_nu;
  real noise_scale;

  real likelihood;

  vector[total_no_measurements] sorted_measurement_times;
  int active_no_measurements;
  real dt;
  int no_fit_sub_steps;
  int no_sim_sub_steps;

  vector[no_persons] centered_VPR;
  vector[no_persons] centered_Fwp;
  vector[no_persons] centered_Fpp;
  vector[no_persons] centered_Ff;
  vector[no_persons] centered_Fl;
  vector[no_persons] centered_Vwp;
  vector[no_persons] centered_Vpp;
  vector[no_persons] centered_Vl;
  vector[no_persons] centered_Pba;
  vector[no_persons] centered_Pwp;
  vector[no_persons] centered_Ppp;
  vector[no_persons] centered_Pf;
  vector[no_persons] centered_Pl;
  vector[no_persons] centered_VMI;
  vector[no_persons] centered_KMI;

  int gq_no_experiments;
  int qg_no_measurements;
  vector[gq_no_experiments] gq_exposures;
  vector[gq_no_experiments] gq_exposure_times;
  vector[gq_no_experiments] gq_alveolar_weights;
  vector[qg_no_measurements] gq_measurement_times;
}
transformed data {
  int no_unconstrained_params = no_latent_params - 2;
  array[no_persons] vector[no_latent_params] centered;
  vector[no_latent_params] log_population_eM_eM = log(population_eM_eM);
  vector[no_latent_params] log_population_eM_eS = log(population_eM_eS);
  vector<lower=0>[no_latent_params] log_population_eS_mu = log(population_eS_mu);
  vector[no_latent_params] log_log_population_eS_mu = log(log_population_eS_mu);
  real active_time_threshold = sorted_measurement_times[active_no_measurements];
  for(person in 1:no_persons){
    centered[person] = [
      centered_VPR[person],
      centered_Fwp[person],
      centered_Fpp[person],
      centered_Ff[person],
      centered_Fl[person],
      centered_Vwp[person],
      centered_Vpp[person],
      centered_Vl[person],
      centered_Pba[person],
      centered_Pwp[person],
      centered_Ppp[person],
      centered_Pf[person],
      centered_Pl[person],
      centered_VMI[person],
      centered_KMI[person]
    ]';
  }
}
parameters {
  vector[no_unconstrained_params] unit_log_population_eM;
  vector[no_latent_params] unit_log_population_eS;
  array[no_persons] vector[no_unconstrained_params] unit_log_person_params;
  vector<lower=0>[2+max(measurement_idxs)] noise;
}
transformed parameters {
  vector[no_latent_params] constrained_unit_log_population_eM = prepare_unit_log_vector(
    unit_log_population_eM
  );
  vector[no_latent_params] log_population_eM = constrain_log_vector(
    log_population_eM_eM + log_population_eM_eS .* constrained_unit_log_population_eM
  );
  vector<lower=0>[no_latent_params] population_eM = exp(log_population_eM);
  vector<lower=0>[no_latent_params] log_population_eS = exp(
      log_log_population_eS_mu + unit_log_population_eS
  );
  vector<lower=1>[no_latent_params] population_eS = exp(
    log_population_eS
  );
  array[no_persons] vector[no_latent_params] constrained_unit_log_person_params;
  array[no_persons] vector[no_latent_params] log_person_params;
  array[no_persons] vector<lower=0>[no_latent_params] person_params;
  real log_likelihood = 0;
  population_eM[6:7] = (
    .837 - population_eM[8]
  ) * population_eM[6:7];
  for(person in 1:no_persons){
    constrained_unit_log_person_params[person] = prepare_unit_log_vector(
      unit_log_person_params[person]
    );
    log_person_params[person] = constrain_log_vector(
      // log_population_eM +
      (1 - centered[person]) .* log_population_eM +
      pow(log_population_eS, 1 - centered[person]) .* constrained_unit_log_person_params[person]
    );
    person_params[person] = exp(log_person_params[person]);
    person_params[person, 6:7] = (
      .837 - person_params[person,8]
    ) * person_params[person, 6:7];
  }
  if(likelihood){
    array[2] vector[total_no_measurements] fit_states;
    int start_idx = 1;
    for(experiment in 1:no_experiments){
      int person = person_idxs[experiment];
      int nidx = no_measurements[experiment];
      int end_idx = start_idx;// + min(active_no_measurements, nidx) - 1;
      // if(measurement_times[start_idx] > active_time_threshold){
      //   continue;
      // }
      for(idx in start_idx:start_idx+nidx-1){
        if(measurement_times[idx] <= active_time_threshold){
          end_idx = idx;
        }else{
          break;
        }
      }
      fit_states[, start_idx:end_idx] = new_simulate_person(
          exposures[experiment],
          exposure_times[experiment],
          measurement_times[start_idx:end_idx],
          person_params[person], measured_params[person],
          dt / no_fit_sub_steps,
          alveolar_weights[experiment]
      );
      for(i in 1:2){
        for(idx in start_idx:end_idx){
          if(is_nan(measurements[i, idx])){
            continue;
          }
          log_likelihood += lognormal_lpdf(
            measurements[i, idx] |
            log(fit_states[i, idx]),
            noise[measurement_idxs[experiment] + i]
          );

        }
      }
      start_idx += nidx;
    }
  }
}
model {
  target += normal_lpdf(constrained_unit_log_population_eM | 0,1);
  target += scaled_inv_chi_square_lpdf(
    pow(log_population_eS, 2) | population_eS_nu, log_population_eS_mu
  );
  target += log(pow(log_population_eS,2));
  for(person in 1:no_persons){
    target += normal_lpdf(
      constrained_unit_log_person_params[person] |
      // 0,
      centered[person] .* pow(log_population_eS, centered[person] - 1) .* log_population_eM,
      pow(log_population_eS, centered[person])
    );
  }
  if(likelihood){
    target += -log(noise);
  }else{
    target += weibull_lpdf(noise/noise_scale | 2,1);
  }
  target += log_likelihood;
}
generated quantities {
  array[no_persons, gq_no_experiments, 2] vector[qg_no_measurements] gq_states;
  for(person in 1:no_persons){
    for(experiment in 1:gq_no_experiments){
      gq_states[person, experiment] = new_simulate_person(
          gq_exposures[experiment],
          gq_exposure_times[experiment],
          gq_measurement_times,
          person_params[person], measured_params[person],
          dt / no_sim_sub_steps,
          gq_alveolar_weights[experiment]
      );
    }
  }
}
