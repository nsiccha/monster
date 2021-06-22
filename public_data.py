import numpy as np

# https://stat.columbia.edu/~gelman/research/published/toxicology.pdf
param_labels = [
    '(VPR)',
    '(Fwp)',
    '(Fpp)',
    '(Ff)',
    '(Fl)',
    '(Vwp)',
    '(Vpp)',
    '(Vl)',
    '(Pba)',
    '(Pwp)',
    '(Ppp)',
    '(Pf)',
    '(Pl)',
    '(VMI)',
    '(KMI)',
]
no_latent_params = len(param_labels)
no_persons = 6
# https://stat.columbia.edu/~gelman/research/published/toxicology.pdf
prior_population_parameters = np.array([
#   [ eM ,  eS ,Sigma, nu, trunc]
    [1.6 , 1.3 , 1.3 , 2, 3],#VPR,0
    [.48 , 1.2, 1.2, 2, 3],#Fwp,1
    [.2  , 1.2, 1.2, 2, 3],#Fpp,2
    [.07 , 1.2, 1.2, 2, 3],#Ff,3
    [.25 , 1.1, 1.1, 2, 3],#Fl,4
    [.28 , 1.2, 1.2, 2, 3],#Vwp,5
    [.56 , 1.2, 1.2, 2, 3],#Vpp,6
    [.033, 1.1 , 1.1 , 2, 3],#Vl,7
    [12  , 1.5 , 1.3 , 2, 3],#Pba,8
    [4.8 , 1.5 , 1.3 , 2, 3],#Pwb,9
    [1.6 , 1.5 , 1.3 , 2, 3],#Ppp,10
    [125 , 1.5 , 1.3 , 2, 3],#Pf,11
    [4.8 , 1.5 , 1.3 , 2, 3],#Pl,12
    [.042, 10  , 2   , 2, 2],#VMI,13
    [16  , 10  , 1.5 , 2, 2]#KMI,14
])

adjusted_prior_population_parameters = prior_population_parameters.copy()
adjusted_prior_population_parameters[1, 0] = .47
adjusted_prior_population_parameters[5, 0] = .27
adjusted_prior_population_parameters[6, 0] = .55
adjusted_prior_population_parameters[1, 1] = 1.17
adjusted_prior_population_parameters[2, 1] = 1.22
adjusted_prior_population_parameters[3, 1] = 1.27
adjusted_prior_population_parameters[4, 1] = 1.15
adjusted_prior_population_parameters[5, 1] = 1.36
adjusted_prior_population_parameters[6, 1] = 1.17
adjusted_prior_population_parameters[1:7, 2] = adjusted_prior_population_parameters[1:7, 1]

# https://stat.columbia.edu/~gelman/research/published/toxicology.pdf
posterior_population_parameters = np.array([
#   [ eM ,  eS ,Sigma, nu, trunc]
    [1.19 , 1.13 , 1.3 , 2, 3],#VPR,0
    [.637 , 1.06, 1.17, 2, 3],#Fwp,1
    [.129  , 1.11, 1.22, 2, 3],#Fpp,2
    [.0488 , 1.12, 1.27, 2, 3],#Ff,3
    [.179 , 1.11, 1.15, 2, 3],#Fl,4
    [.196 , 1.09, 1.36, 2, 3],#Vwp,5
    [.641 , 1.03, 1.17, 2, 3],#Vpp,6
    [.033, 1.04 , 1.1 , 2, 3],#Vl,7
    [16.0  , 1.11 , 1.3 , 2, 3],#Pba,8
    [1.92 , 1.12 , 1.3 , 2, 3],#Pwb,9
    [2.90 , 1.15 , 1.3 , 2, 3],#Ppp,10
    [84.1 , 1.28 , 1.3 , 2, 3],#Pf,11
    [3.08 , 1.12 , 1.3 , 2, 3],#Pl,12
    [.00191, 1.45  , 2   , 2, 2],#VMI,13
    [.729  , 1.2  , 1.5 , 2, 2]#KMI,14
])
posterior_population_parameters[:, 2] = posterior_population_parameters[:, 1]
posterior_population_parameters[:, 3] = 2+no_persons
posterior_population_parameters[:, 4] = 3

# https://stat.columbia.edu/~gelman/research/published/toxicology.pdf
posterior_person_parameters = np.array([
    [#VPR,0
        [1.16, 1.26, 1.19, 1.33, 1.22, .961],
        [1.15, 1.15, 1.14, 1.15, 1.15, 1.15]
    ],
    [#Fwp,1
        [.653, .658, .647, .660, .626, .606],
        [1.06, 1.07, 1.07, 1.06, 1.08, 1.08]
    ],
    [#Fpp,2
        [.121, .123, .127, .123, .132, .134],
        [1.12, 1.13, 1.13, 1.12, 1.13, 1.13]
    ],
    [#Ff,3
        [.048, .0442, .0462, .0437, .0507, .0582],
        [1.13, 1.13, 1.14, 1.13, 1.14, 1.14]
    ],
    [#Fl,4
        [.173, .170, .175, .168, .185, .195],
        [1.15, 1.16, 1.15, 1.15, 1.16, 1.15]
    ],
    [#Vwp,5
        [.189, .201, .202, .201, .183, .188],
        [1.14, 1.15, 1.15, 1.15, 1.15, 1.14]
    ],
    [#Vpp,6
        [.649, .636, .636, .636, .655, .65],
        [1.04, 1.05, 1.05, 1.05, 1.04, 1.04]
    ],
    [#Vl,7
        [.032, .033, .033, .033, .033, .032],
        [1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    ],
    [#Pba,8
        [15.1, 16.4, 15.3, 15.6, 18.7, 15.8],
        [1.04, 1.03, 1.04, 1.04, 1.04, 1.04]
    ],
    [#Pwp,9
        [1.83, 1.98, 1.95, 2.00, 1.83, 1.83],
        [1.15, 1.16, 1.16, 1.16, 1.15, 1.14]
    ],
    [#Ppp,10
        [2.94, 2.59, 2.51, 2.76, 4.06, 2.96],
        [1.08, 1.09, 1.09, 1.08, 1.09, 1.09]

    ],
    [#Pf,11
        [82.3, 69.1, 73.9, 49.1, 171, 85.4],
        [1.08, 1.08, 1.08, 1.08, 1.09, 1.07]

    ],
    [#Pl,12
        [2.93, 3.07, 3.21, 3.09, 3.16, 2.94],
        [1.32, 1.33, 1.32, 1.33, 1.33, 1.32]

    ],
    [#VMI,13
        [.0011, .00139, .00214, .00199, .00415, .00165],
        [1.41, 1.37, 1.30, 1.34, 1.30, 1.38]

    ],
    [#KMI,14
        [.801, .754, .660, .742, .650, .771],
        [1.63, 1.61, 1.59, 1.57, 1.59, 1.60]
    ]
])

# https://stat.columbia.edu/~gelman/research/published/toxicology.pdf
measured_params = np.array([
    [62, 71, 71, 74, 61, 61], #lean_body_mass
    [.114, .134, .134, .14, .09, .208], #prop_mass_fat
    [7.6, 11.6, 10, 11.3, 12.3, 8.8], #Flow pul.
]).T

# https://www.gnu.org/software/mcsim/mcsim.html#perc_002emodel
ppm_to_mgl = .488/72

ppm_exposures = np.array([72,144])
exposures = ppm_exposures*ppm_to_mgl #exposure concentration

def get_base_data(population_parameters, std_trunc, pop_trunc, person_trunc):
    no_latent_params = len(population_parameters)

    population_truncation = [
        -population_parameters[:,4]-pop_trunc,
        +population_parameters[:,4]+pop_trunc
    ]
    std_truncation = [
        np.full(no_latent_params, -np.inf),
        np.full(no_latent_params, std_trunc)
    ]
    person_truncation = [
        population_truncation[0]-person_trunc,
        population_truncation[1]+person_trunc
    ]

    return dict(
        no_latent_params=no_latent_params,
        population_eM_eM=population_parameters[:, 0],
        population_eM_eS=population_parameters[:, 1],
        population_eS_mu=population_parameters[:, 2],
        population_eS_nu=population_parameters[:, 3],
        population_truncation=population_truncation,
        std_truncation=std_truncation,
        person_truncation=person_truncation,
    )
