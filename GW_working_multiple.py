#!/home/mattia.emma/.conda/envs/sbi_test/bin/python

import sys
import time
from collections import namedtuple
import json
import argparse
import os
import bilby
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

import bilby.core.likelihood.simulation_based_inference as sbibilby
from bilby.core.likelihood.simulation_based_inference import GenerateData
import matplotlib.pyplot as plt
from bilby.gw.likelihood.simulation_based_inference import GenerateWhitenedIFONoise
from bilby.gw.likelihood.simulation_based_inference import GenerateWhitenedSignal

class BenchmarkLikelihood(object):
    def __init__(
        self,
        benchmark_likelihood,
        reference_likelihood,
        prior,
        reference_prior,
        outdir,
        injection_parameters,
    ):
        self.benchmark_likelihood = benchmark_likelihood
        self.reference_likelihood = reference_likelihood
        self.prior = prior
        self.reference_prior = reference_prior
        self.outdir = outdir
        self.injection_parameters = injection_parameters
        self.statistics = dict(
            likelihood_class=benchmark_likelihood.__class__.__name__,
            likelihood_metadata=benchmark_likelihood.meta_data,
        )

    def _time_likelihood(self, likelihood, n, name):
        eval_times = []
        for _ in range(n):
            likelihood.parameters.update(self.prior.sample())
            start = time.time()
            likelihood.log_likelihood()
            end = time.time()
            eval_times.append(end - start)
        self.statistics[f"likelihood_{name}_eval_time_mean"] = float(
            np.mean(eval_times)
        )
        self.statistics[f"likelihood_{name}_eval_time_std"] = float(np.std(eval_times))

    def benchmark_time(self, n=100):
        self._time_likelihood(self.benchmark_likelihood, n, "benchmark")
        self._time_likelihood(self.reference_likelihood, n, "reference")

    def benchmark_posterior_sampling(self, run_sampler_kwargs=None):
        kwargs = dict(nlive=1000, sampler="dynesty", dlogz=0.5, check_point_delta_t=60)
        if run_sampler_kwargs is not None:
            kwargs.update(run_sampler_kwargs)

        result_reference = bilby.run_sampler(
            likelihood=self.reference_likelihood,
            priors=self.reference_prior,
            outdir=self.outdir,
            injection_parameters=injection_parameters,
            label=self.benchmark_likelihood.label + "_REFERENCE",
            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            **kwargs,
        )
        
        result_benchmark = bilby.run_sampler(
            likelihood=self.benchmark_likelihood,
            priors=self.prior,
            outdir=self.outdir,
            injection_parameters=injection_parameters,
            label=self.benchmark_likelihood.label,
            **kwargs,
        )

        for key, val in self.reference_prior.items():
            if val.is_fixed is False:
                samplesA = result_benchmark.posterior[key]
                samplesB = result_reference.posterior[key]
                js = calculate_js(samplesA, samplesB)
                self.statistics[f"1D_posterior_JS_{key}_median"] = js.median
                self.statistics[f"1D_posterior_JS_{key}_plus"] = js.plus
                self.statistics[f"1D_posterior_JS_{key}_minus"] = js.minus

                fig, ax = plt.subplots()
                ax.hist(samplesA, bins=50, alpha=0.8, label="Benchmark")
                ax.hist(samplesB, bins=50, alpha=0.8, label="Reference")
                ax.axvline(self.injection_parameters[key], color="k")
                ax.set(xlabel=key, title=f"JS={js.median}")
                ax.legend()
                plt.savefig(
                        f"{self.outdir}/{self.benchmark_likelihood.label}_1D_posterior_{key}.png"
                )
                    

    def write_results(self):
        bilby.utils.check_directory_exists_and_if_not_mkdir("RESULTS_working")
        with open(
            f"RESULTS_working/result_benchmark_{self.benchmark_likelihood.label}.json", "w"
        ) as file:
            json.dump(self.statistics, file, indent=4)

def calc_summary(jsvalues, quantiles=(0.16, 0.84)):
    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(jsvalues, quants_to_compute * 100)
    summary = namedtuple("summary", ["median", "lower", "upper"])
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]
    return summary


def calculate_js(samplesA, samplesB, ntests=100, xsteps=100):
    js_array = np.zeros(ntests)
    for j in range(ntests):
        nsamples = min([len(samplesA), len(samplesB)])
        A = np.random.choice(samplesA, size=nsamples, replace=False)
        B = np.random.choice(samplesB, size=nsamples, replace=False)
        xmin = np.min([np.min(A), np.min(B)])
        xmax = np.max([np.max(A), np.max(B)])
        x = np.linspace(xmin, xmax, xsteps)
        A_pdf = gaussian_kde(A)(x)
        B_pdf = gaussian_kde(B)(x)

        js_array[j] = np.nan_to_num(np.power(jensenshannon(A_pdf, B_pdf), 2))

    return calc_summary(js_array)






print(f"Running command {' '.join(sys.argv)}")

parser = argparse.ArgumentParser()
parser.add_argument("--dimensions", type=int,default=1)
parser.add_argument("--likelihood", type=str)
parser.add_argument("--num-simulations", type=int)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--nlive", type=int, default=1000)
parser.add_argument("--dlogz", type=float, default=0.5)
parser.add_argument("--rseed", type=int, default=42)
parser.add_argument("--time_lower", type=float, default=0)
parser.add_argument("--time_upper", type=float, default=0)
args = parser.parse_args()

outdir = "outdir_benchmark_gw/Runs_working_L_"+args.likelihood+'_N'+str(args.num_simulations)+'_D'+str(args.dimensions)+'_TL'+str(args.time_lower)+'_TU'+str(args.time_upper)
np.random.seed(args.rseed)
times=[args.time_lower, args.time_upper]
num_simulations = args.num_simulations
################################################# Old code ######################################################
injection_parameters = dict(
    chirp_mass=30.0,
    mass_ratio=1.0,
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

#noise_parameters=dict( sigma=1,)

signal_priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "mass_ratio",
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
signal_priors['chirp_mass']=bilby.gw.prior.UniformInComponentsChirpMass(minimum=20, maximum=40, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)
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


############################################Use this new yobs######################################################
yobs = ifo.whitened_time_domain_strain
xobs = ifo.time_array

full_noise = GenerateWhitenedIFONoise(ifo, False, times)
full_signal = GenerateWhitenedSignal(ifo, waveform_generator, signal_priors, False, times)
full_signal_and_noise = sbibilby.AdditiveSignalAndNoise(full_signal, full_noise)

priors = noise_priors | signal_priors
priors = bilby.core.prior.PriorDict(priors)
priors.convert_floats_to_delta_functions()


##################### New code#########################################

#Training the neural network 
label = f"SG_{args.likelihood}_D{args.dimensions}_N{num_simulations}_R{args.repeat}"
if args.likelihood == "NLE":
    benchmark_likelihood = sbibilby.NLELikelihood(
        yobs,
        full_signal_and_noise,
        bilby_prior=priors,
        label=label,
        num_simulations=num_simulations,
        cache_directory=outdir,
        show_progress_bar=True,
        cache=True,
    )
elif args.likelihood == "RNLE":
    benchmark_likelihood = sbibilby.NLEResidualLikelihood(
        yobs,
        full_signal_and_noise,
        bilby_prior=priors,
        label=label,
        num_simulations=num_simulations,
        cache_directory=outdir,
    )
start=time.time()
benchmark_likelihood.init()
end=time.time()
total_time=end-start
file_path = "RESULTS_working_times/Training_time_benchmark.txt"
if not os.path.exists('RESULTS_working_times'):
    # Create the directory
    os.mkdir('RESULTS_working')
# Check if the file exists
if not os.path.exists(file_path):
    # Create the file if it does not exist
    with open(file_path, "w") as f:
        pass  # Simply creating the file
use_mask=False
with open(file_path, "a") as f:
    f.write(f"{total_time:.2f} {num_simulations} {use_mask} {args.likelihood} {times[0]} {times[1]}\n")
    f.close()

############################################ Reference likelihood #####################################
reference_injection_parameters = dict(
    chirp_mass=30.0,
    mass_ratio=1.0,
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

reference_signal_priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "mass_ratio",
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
    reference_signal_priors[key] = reference_injection_parameters[key]
reference_signal_priors['chirp_mass']=bilby.gw.prior.UniformInComponentsChirpMass(minimum=20, maximum=40, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)


duration = 4.0
sampling_frequency = 1024.0
minimum_frequency = 20
trigger = 1126259642.4
start_time = trigger - duration / 2

reference_waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)


reference_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=reference_waveform_arguments,
)

reference_ifos = bilby.gw.detector.InterferometerList(['H1'])
reference_ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)
reference_ifos.inject_signal(
    waveform_generator=reference_waveform_generator, parameters=reference_injection_parameters
)


reference_priors = reference_signal_priors
reference_priors = bilby.core.prior.PriorDict(reference_priors)
reference_priors.convert_floats_to_delta_functions()


reference_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    reference_ifos,
    reference_waveform_generator,
    priors=reference_priors,
 )
######################################   Benchmark ##################################
bench = BenchmarkLikelihood(
    benchmark_likelihood,
    reference_likelihood,
    priors,
    reference_priors,
    outdir,
    injection_parameters=reference_injection_parameters,
    
)
bench.benchmark_time()
bench.benchmark_posterior_sampling(
    dict(
        sampler="dynesty",
        nlive=args.nlive,
        dlogz=args.dlogz,
        resume=args.resume,
        print_method="interval-10",
        npool=8,
        sample="acceptance-walk",
        check_point_delta_t=180,
    )
)
bench.write_results()

######################################### Old code ####################################################################



#result=bilby.core.result.read_in_result(filename='outdir_2000/TEST_2000_result.json')
#truths = {'sigma':1.0,'chirp_mass':28.1}
#result.plot_corner(truths=truths)