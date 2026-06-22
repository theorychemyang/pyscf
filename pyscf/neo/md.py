#!usr/bin/env python3
'''
calculate power spectrum and IR spectrum based on MD trajectory
'''
import math
import scipy
import numpy
from scipy import signal
from ase.io.trajectory import Trajectory
from ase import units
from pyscf import lib

def _cumulative_trapezoid(y, x):
    if hasattr(scipy.integrate, 'cumulative_trapezoid'):
        return scipy.integrate.cumulative_trapezoid(y, x)
    return scipy.integrate.cumtrapz(y, x)

def step_average(data):
    'get cumulative averaged data'
    return numpy.cumsum(data)/numpy.arange(1, len(data) + 1)

def calc_ACF(traj):
    'calculate auto-correlation functions'
    autocorr = signal.correlate(traj, traj, mode='full')[len(traj)-1:]
    autocorr = numpy.multiply(autocorr, 1 / numpy.arange(len(traj), 0, -1))
    return autocorr

def hann(length):
    'Hann window function'
    n = numpy.arange(length)
    return numpy.power(numpy.cos(math.pi*n/(2*(length-1))), 2)

def vacf(datafile, start=0, end=-1, step=1):
    'calculate verlocity auto-correlation function (VACF) from trajectory of MD simulations'
    traj = Trajectory(datafile)

    v = []
    for i in range(len(traj)):
        v.append(traj[i].get_velocities())
    v = numpy.array(v)

    t, natoms, x = v.shape

    mass = traj[-1].get_masses()
    acf = 0

    for i in range(natoms): # i-th atom
        for j in range(x): # j-th component of verlocity
            acf += calc_ACF(v[start:end:step,i,j]) * mass[i]

    return lib.tag_array(acf, natoms=natoms)

def dacf(datafile, time_step=0.5, start=0, end=-1, step=1):
    'calculate dipole auto-correlation function (DACF) from trajectory of MD simulations'
    traj = Trajectory(datafile)
    natoms = len(traj[-1].get_positions())

    dipole = []
    for i in range(len(traj)):
        dipole.append(traj[i].get_dipole_moment())
    dipole = numpy.array(dipole)

    t, x = dipole.shape

    acf = 0

    for j in range(x):
        de = numpy.gradient(dipole[start:end:step, j], time_step)
        acf += calc_ACF(de)

    return lib.tag_array(acf, natoms=natoms)

def calc_FFT(acf):
    'get Fourier transform of ACF'

    # window function
    acf *= hann(len(acf))

    # zero padding
    N = 2 ** (math.ceil(math.log(len(acf), 2)) + 2)
    acf = numpy.append(acf, numpy.zeros(N - len(acf)))

    # data mirroring
    acf = numpy.concatenate((acf, acf[::-1][:-1]), axis=0)

    yfft = numpy.fft.fft(acf, axis=0)

    #numpy.set_printoptions(threshold=numpy.inf)
    #print(yfft)

    return yfft

def spectrum(acf, time_step=0.5, corr_depth=4096, natoms=None):
    'get wavenumber and intensity of spectrum'

    if natoms is None:
        natoms = getattr(acf, 'natoms', None)
    if natoms is None:
        raise ValueError('natoms must be provided when acf has no natoms tag')

    acf = acf[:corr_depth]

    yfft = calc_FFT(acf)

    fs2cm = 1e-15 * units._c * 100

    wavenumber = numpy.fft.fftfreq(len(yfft), time_step * fs2cm)[0:int(len(yfft)/2)]
    intensity = numpy.real(yfft[0:int(len(yfft)/2)])
    factor = 11604.52500617 * fs2cm / (3 * natoms) # eV2K * fs2cm -> K*cm
    intensity *= factor
    temperature = _cumulative_trapezoid(intensity, wavenumber)
    print('Integrated temperature (K)', temperature[-1])
    return wavenumber, intensity, temperature

def gen_init_vel(modes, masses, temperature=300):
    '''
    Generate initial velocites for MD simulations.

    modes: normalized eigenvectors from mass-weighted Hessian matrix

    Ref:  J. Chem. Theory Comput. 2023, 19, 9358-9368
    '''
    n, m = modes.shape
    natoms = int(n/3)
    indices = numpy.asarray(range(natoms))

    im = numpy.repeat(masses[indices]**-0.5, 3)

    modes = numpy.einsum('in,n->in', modes, im)

    scalar = numpy.sqrt(2*units.kB/units.J*temperature/units._amu*units.m**2) #amu**0.5 ang s**-1
    phi = (2*numpy.random.randint(0, 2, size=(3*natoms-6))-1) # randomize velocity direction +1 or -1

    v = numpy.einsum('i,in->n', phi, modes[6:,:])*scalar/units.second

    return v.reshape((natoms,3))
