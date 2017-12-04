from numba import jit
from multiprocessing import Pool, Queue
import sys
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import time
from struct import pack, calcsize

NUM_PROCS = 4

SHOW_FILTERS = False
SHOW_PROGRESS = True
SHOW_RESULT = True

POIS_LAMBDA = 0.01
FIRING_DELAY = 960
SAMPLE_RATE = 96000
PAD_SIZE = 9600
WINDOW_SIZE = 9600
NUM_IHCS = 3500
SIM_IHCS = (0, 3500)

# read in signal
s_in = wavfile.read("andy_96k.wav")[1] / np.iinfo(np.int32).max
#s_in = wavfile.read("f480_gliss.wav")[1] / np.iinfo(np.int32).max
 
@jit
def fourth_dapgf(Q, w):
  # fourth order DAPGF, continuous-time
  w = np.float64(w)
  Q = np.float64(Q)

  term1 = w / Q

  # fourth order DAPGF
  b = np.array((np.power(w, 7.0), 0.0), dtype=np.float64)

  a = np.array((1.0,
                4*term1,
                6*np.power(term1, 2) + 4*np.power(w, 2),
                4*np.power(term1, 3) + 12*term1*np.power(w, 2),
                np.power(term1, 4) + 12*np.power(term1, 2)*np.power(w, 2) + 6*np.power(w, 4),
                4*np.power(term1, 3)*np.power(w, 2) + 12*term1*np.power(w, 4),
                6*np.power(term1, 2)*np.power(w, 4) + 4*np.power(w, 6),
                4*term1*np.power(w, 6),
                np.power(w, 8)), dtype=np.float64)

  return b, a

@jit
def first_apgf(Q, w):
  # first order APGF, continuous-time
  b = np.array((np.power(w, 2)), dtype=np.float64)
  a = np.array((1.0, w / Q, np.power(w, 2)), dtype=np.float64)

  return signal.normalize(b, a)

@jit
def bp_biquad(Q, w0, Fs):
  # http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

  w0 = w0/Fs
  alpha = np.sin(w0)/(2*Q)

  b0 = Q*alpha
  b1 = 0 
  b2 = -Q*alpha
  a0 = 1 + alpha
  a1 = -2*np.cos(w0)
  a2 = 1 - alpha

  b, a = signal.normalize((b0, b1, b2), (a0, a1, a2))

  return np.concatenate((b, a))

@jit
def calc_discrete_filter(i):
  ## method 1: cont2discrete
  coeffs = signal.cont2discrete(first_apgf(gain[i], cf_rad[i]), dt=1.0/SAMPLE_RATE, method="bilinear")
  # don't know why cont2discrete outputs first term as a nested list
  coeffs = (coeffs[0][0], coeffs[1])
  # generate bank of three LP biquads (aka 3rd order APGF)
  filter_bank[i] = np.tile(np.concatenate(coeffs), (4, 1))
  # and add on a BP biquad to make a fourth order DAPGF
  filter_bank[i][3] = bp_biquad(10.0, cf_warp[i], SAMPLE_RATE)

# build filter bank
cont_filter_bank = {}
filter_bank = np.zeros((NUM_IHCS, 4, 6))
zi = np.zeros((NUM_IHCS, 4, 2))
cf = {}
cf_rad = {}
cf_warp = {}
gain = np.zeros(NUM_IHCS)

sys.stderr.write("calculating coefficients...\n")
s_time = time.time()

if SHOW_FILTERS:
  plt.ion()
  plt.subplots_adjust(hspace=1.0)

for i in range(*SIM_IHCS):
  # start with gain (q) of 1/2**0.5 (minimum for 0 dB gain)
  #gain[i] = 0.5**0.5
  gain[i] = 2.6

  # greenwood function
  # https://en.wikipedia.org/wiki/Greenwood_function
  cf[i] = 165.4*(10**(2.1*(i+1.0)/NUM_IHCS)-0.88)
  cf_rad[i] = 2*np.pi*cf[i]

  # must compensate for warping
  # https://en.wikipedia.org/wiki/Bilinear_transform#Frequency_warping
  cf_warp[i] = (2.0*SAMPLE_RATE) * np.arctan((0.5/SAMPLE_RATE)*cf_rad[i])

  # 4th order DAPGF, analog, only for plotting
  cont_filter_bank[i] = fourth_dapgf(gain[i], cf_rad[i])

  # this is the filter we'll actually use
  calc_discrete_filter(i)

  # setup initial conditions for lfilter
  zi[i] = np.tile([0, 0], (4, 1))

  if SHOW_FILTERS and i % 10 == 0:
    plt.clf()

    # frequency response of continuous time filter
    w, mag, phase = signal.bode(cont_filter_bank[i], w=2*np.pi*np.linspace(0.0, 22050, num=2000))

    plt.subplot(3, 1, 1)
    plt.title("Freq. Resp. Analog CF=%.02f" % cf[i])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("dB")
    plt.ylim(-30, 100)
    plt.semilogx(w/(2*np.pi), mag)

    # frequency response of discrete time filter
    w, h = signal.sosfreqz(filter_bank[i], 1500)

    plt.subplot(3, 1, 2)
    plt.title("Freq. Resp. Digital CF=%.02f" % cf[i])
    plt.xlabel("Normalized Frequency (rad/sample)")
    plt.ylabel("dB")
    plt.ylim(-30, 100)
    plt.semilogx(w/np.pi, 20*np.log10(np.abs(h)))

    # plot of testing discrete time filter
    signal_in = np.sin(2*np.pi*cf[i]*np.linspace(0, 0.1, SAMPLE_RATE*0.1)) + 0.5*np.random.randn(int(SAMPLE_RATE*0.1))
    signal_out = signal.sosfilt(filter_bank[i], x=signal_in)

    ax1 = plt.subplot(3, 1, 3)
    plt.title("Filter Test CF=%.02f" % cf[i])
    plt.xlabel("Sample")

    ax1.plot(signal_in)
    ax2 = ax1.twinx()
    ax2.plot(signal_out, color="orange")

    plt.pause(0.01)

  sys.stderr.write("\rihc: %s cf: %.02f cf_rad: %.02f cf_warp: %.02f" %
                   (i+1, cf[i], cf_rad[i], cf_warp[i]))

if SHOW_FILTERS:
  plt.ioff()

sys.stderr.write("\ntook %.02f seconds\n" % (time.time() - s_time,))

# filter using filterbank
sys.stderr.write("first pass: filtering waveform...\n")

if SHOW_PROGRESS:
  plt.ion()

# process entire waveform in one pass
s_time = time.time()

output_bank = np.zeros((NUM_IHCS, len(s_in)))

# use multiprocessing to speed this up
def _filt_sig(q_in, q_out):
  while True:
    i, coeffs, zi = q_in.get(True)
    s_out, zf = signal.sosfilt(coeffs, x=padded_data, zi=zi)
    q_out.put((i, s_out))

padded_data = np.concatenate((s_in[:PAD_SIZE], s_in[:PAD_SIZE][::-1], s_in))

mp_q_in = Queue()
mp_q_out = Queue()
mp_p = Pool(NUM_PROCS, _filt_sig, (mp_q_in, mp_q_out))

for i in range(*SIM_IHCS):
  mp_q_in.put((i, filter_bank[i], zi[i]))

for _ in range(*SIM_IHCS):
  i, s_out = mp_q_out.get(True)
  s_out = s_out[2*PAD_SIZE:]
  output_bank[i] = s_out

  if SHOW_PROGRESS and i % 100 == 0:
    plt.clf()
    plt.title("IHC CF: %.02f" % cf[i])
    plt.xlabel("Time (sample)")
    plt.ylabel("Amplitude (a.u.)")

    ax1 = plt.subplot()
    ax1.plot(s_in)

    ax2 = ax1.twinx()
    ax2.plot(s_out, color="orange")

    plt.pause(0.001)

  sys.stderr.write("\rihc: %s cf: %.02f" % (i+1, cf[i]))

sys.stderr.write("\ntook %.02f seconds\n" % (time.time() - s_time))

mp_p.terminate()

if SHOW_PROGRESS:
  plt.ioff()

@jit(nopython=True)
def find_first_zero(vec):
  for i in range(len(vec)):
    if vec[i] == 0:
      return i

  raise ValueError("No zero found in array")

@jit
def calc_firings(t_range):
  timeout = np.zeros(NUM_IHCS, dtype=np.int64)
  time_since_last = np.zeros(NUM_IHCS, dtype=np.int64)
  firings = np.zeros((NUM_IHCS, int(len(t_range) / FIRING_DELAY)), dtype=np.uint32)

  for t in t_range:
    if t % 100 == 0:
      sys.stderr.write("\rtime %d of %d" % (t, len(t_range)))
  
    for i in np.where(np.random.poisson(0.01, NUM_IHCS))[0]:
      if timeout[i] == 0 and \
         output_bank[i][t] >= 1.0 and \
         output_bank[i][t-1] < 1.0:
        if time_since_last[i] > 2**32:
          firings[i][find_first_zero(firings[i])] = 2**32
        else:
          firings[i][find_first_zero(firings[i])] = time_since_last[i]
          time_since_last[i] = 0
  
        timeout[i] = FIRING_DELAY
  
    timeout = np.subtract(timeout, 1).clip(0)
    time_since_last = np.add(time_since_last, 1)

  sys.stderr.write("\n")

  return firings

def calc_rates(firings):
  rates = np.zeros(NUM_IHCS)

  for i in range(*SIM_IHCS):
    rates[i] = np.mean(firings[i][firings[i].nonzero()])

  return rates

def adjust_gain(rates):
  filters_to_recalc = np.zeros(NUM_IHCS)

  for i in range(*SIM_IHCS):
    # if rate is low, turn up gain slowly
    if rates[i] > FIRING_DELAY:
      gain[i] *= np.exp(0.1)
      calc_discrete_filter(i)
      filters_to_recalc[i] = 1
    # if rate is high, turn gain down quickly
    elif rates[i] < FIRING_DELAY:
      gain[i] *= np.exp(-0.2)
      calc_discrete_filter(i)
      filters_to_recalc[i] = 1

  return np.where(filters_to_recalc == 1)

# determine when neurons are activated and adjust gain
sys.stderr.write("second pass: converting to naive neural encoding...\n")
s_time = time.time()

firings = calc_firings(range(0, len(s_in)))

sys.stderr.write("\ntook %.02f seconds\n" % (time.time() - s_time,))

if SHOW_RESULT:
  coding_x = []
  coding_y = []

  for i in range(*SIM_IHCS):
    i_firings = firings[i].nonzero()[0]

    if len(i_firings) > 0:
      for j in i_firings:
        if j == 0:
          coding_x.append(firings[i][0])
        else: 
          coding_x.append(firings[i][:j].sum())
        coding_y.append(cf[i])

  plt.figure()
  plt.title("Cochlear Encoding")
  plt.xlabel("Time (sample)")
  #plt.ylabel("IHC CF")
  #plt.scatter(coding_x, coding_y, s=2)
  plt.ylabel("log2(IHC CF)")
  plt.scatter(coding_x, np.log2(coding_y), s=2)
  plt.show()

with open("sparse.txt", "wb") as fp:
  # header of file contains size of float for IHC CF,
  # size of unsigned int used for number of firings per IHC,
  # and size of unsigned int used for time between firings
  # in the first three bytes
  fp.write(pack("B", calcsize("f"))) # they are all 32-bits
  fp.write(pack("B", calcsize("I")))
  fp.write(pack("B", calcsize("I")))

  for i in range(*SIM_IHCS):
    i_firings = firings[i].nonzero()[0]

    if len(i_firings) > 0:
      # per-IHC header contains CF (float) and number of firings (uint)
      fp.write(pack("f", round(cf[i], 6)))
      fp.write(pack("I", len(i_firings)))

      for j in i_firings:
        fp.write(pack("I", firings[i][j]))
