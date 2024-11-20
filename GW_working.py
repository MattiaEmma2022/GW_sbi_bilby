import bilby
import numpy as np
import bilby.core.likelihood.simulation_based_inference as sbibilby
from bilby.core.likelihood.simulation_based_inference import GenerateData
import matplotlib.pyplot as plt
from bilby.gw.likelihood.simulation_based_inference import GenerateWhitenedIFONoise
from bilby.gw.likelihood.simulation_based_inference import GenerateWhitenedSignal

#Create priors and signal
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=1000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

#injection_parameters['sigma'] = 1

signal_priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
    "luminosity_distance",
    "theta_jn"
]:
    signal_priors[key] = injection_parameters[key]
signal_priors['mass_ratio']=29.0/36.0
noise_priors = bilby.core.prior.PriorDict(dict(sigma=bilby.core.prior.Uniform(0, 2, 'sigma')))

duration = 4.0
sampling_frequency = 1024.0
minimum_frequency = 20
trigger = 1126259642.4
start_time = trigger - duration / 2

waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

outdir = "outdir_1000"
label = "TEST_1000"

ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)
ifo = ifos[0]

yobs = ifo.whitened_time_domain_strain
xobs = ifo.time_array
#Mask data
window_start = trigger - 0.5
window_end = trigger + 0.2
mask = (xobs >= window_start) & (xobs <= window_end)

# Slice time and data arrays
xobs = xobs[mask]
yobs = yobs[mask]

#Create the simulated data from noise and signal simulation
noise = GenerateWhitenedIFONoise(ifo)
signal = GenerateWhitenedSignal(ifo, waveform_generator, signal_priors)
signal_and_noise = sbibilby.AdditiveSignalAndNoise(signal, noise)

priors = noise_priors | signal_priors
priors = bilby.core.prior.PriorDict(priors)
priors.convert_floats_to_delta_functions()
sample = priors.sample()
signal_and_noise.get_data(sample)

#Training the neural network

likelihood = sbibilby.NLELikelihood(yobs, signal_and_noise, priors, label, show_progress_bar=True, num_simulations=1000)
likelihood.init()

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    npoints=250,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    check_point_delta_t=60,
    #conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)


result=bilby.core.result.read_in_result(filename='outdir_2000/TEST_2000_result.json')
truths = {'sigma':1.0,'chirp_mass':28.1}
result.plot_corner(truths=truths)