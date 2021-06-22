data {
  int no_persons;
  int no_latent_params;
  vector<lower=0>[no_latent_params] population_eM_eM;
  vector<lower=0>[no_latent_params] population_eM_eS;
  vector<lower=1>[no_latent_params] population_eS_mu;
  vector<lower=0>[no_latent_params] population_eS_nu;
  array[2] vector[no_latent_params] population_truncation;
  array[2] vector[no_latent_params] std_truncation;
  array[2] vector[no_latent_params] person_truncation;
  real noise_scale;

  real likelihood;
  real observed_states[no_persons, no_latent_params];
}
transformed data {
  vector[no_latent_params] log_population_eM_eS = log(population_eM_eS);
  vector<lower=0>[no_latent_params] log_population_eS_mu = log(population_eS_mu);
  vector[no_latent_params] log_log_population_eS_mu = log(log_population_eS_mu);
}
parameters {
  vector<
    lower=population_truncation[1],
    upper=population_truncation[2]
  >[no_latent_params] unit_log_population_eM;
  vector<
    lower=std_truncation[1],
    upper=std_truncation[2]
  >[no_latent_params] unit_log_population_eS;
  array[no_persons] vector<
    lower=rep_array(
      (
        (person_truncation[1] - unit_log_population_eM) .* log_population_eM_eS
      ) ./ exp(
        log_log_population_eS_mu + unit_log_population_eS
      ),
      no_persons
    ),
    upper=rep_array(
      (
        (person_truncation[2] - unit_log_population_eM) .* log_population_eM_eS
      ) ./ exp(
        log_log_population_eS_mu + unit_log_population_eS
      ),
      no_persons
    )
  >[no_latent_params] unit_log_person_params;
  real<lower=0> noise;
}
transformed parameters {
  vector<lower=0>[no_latent_params] population_eM = (
      population_eM_eM .* pow(population_eM_eS, unit_log_population_eM)
  );
  vector<lower=0>[no_latent_params] log_population_eS = exp(
      log_log_population_eS_mu + unit_log_population_eS
  );
  vector<lower=1>[no_latent_params] population_eS = exp(
    log_population_eS
  );
  array[no_persons] vector<lower=0>[no_latent_params] person_params;
  population_eM[2:5] = population_eM[2:5] / sum(population_eM[2:5]);
  population_eM[6:7] = (
    .837 - population_eM[8]
  ) * population_eM[6:7] / sum(population_eM[6:7]);
  for(person in 1:no_persons){
    person_params[person] = (
      population_eM .* pow(population_eS, unit_log_person_params[person])
    );
    person_params[person, 2:5] = person_params[person, 2:5] / sum(person_params[person, 2:5]);
    person_params[person,6:7] = (
      .837 - person_params[person,8]
    ) * person_params[person,6:7] / sum(person_params[person,6:7]);
  }
}
model {
  for(person in 1:no_persons){
    target += normal_lpdf(unit_log_person_params[person] | 0,1);
  }
  target += normal_lpdf(unit_log_population_eM | 0,1);
  target += scaled_inv_chi_square_lpdf(
    pow(log_population_eS, 2) | population_eS_nu, log_population_eS_mu
  );
  target += log(pow(log_population_eS,2));
  if(likelihood){
    target += -log(noise);
  }else{
    target += weibull_lpdf(noise/noise_scale | 2,1);
  }
  if(likelihood){
    for(person in 1:no_persons){
      target += lognormal_lpdf(observed_states[person] | log(person_params[person]), noise);
    }
  }
}
generated quantities {
  real predicted_states[no_persons, no_latent_params];
  for(person in 1:no_persons){
    predicted_states[person] = lognormal_rng(log(person_params[person]), noise);
  }
}
