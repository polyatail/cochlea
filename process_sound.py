from multiprocessing import Pool, Queue
import sys
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import time

NUM_PROCS = 4

SHOW_FILTERS = False
SHOW_PROGRESS = False
SHOW_RESULT = True

POIS_LAMBDA = 0.01
SAMPLE_RATE = 96000
CHUNK_SIZE = 9600
PAD_SIZE = 9600
NUM_IHCS = 3500
SIM_IHCS = (0, 3500)

# read in signal
s_in = wavfile.read("andy_96k.wav")[1] / np.iinfo(np.int32).max

def first_apgf(Q, w):
  # first order APGF, continuous-time
  b = np.array((np.power(w, 2)), dtype=np.float64)
  a = np.array((1.0, w / Q, np.power(w, 2)), dtype=np.float64)

  return signal.normalize(b, a)

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

# build filter bank
cont_filter_bank = {}
filter_bank = {}
zi = {}
cf = {}
cf_rad = {}
cf_warp = {}
gain = {}

sys.stderr.write("calculating coefficients...\n")
s_time = time.time()

if SHOW_FILTERS:
  plt.ion()
  plt.subplots_adjust(hspace=1.0)

for i in range(*SIM_IHCS):
  # start with gain (q) of 1/2**0.5 (minimum for 0 dB gain)
  #gain[i] = 0.5**0.5
  gain[i] = 2.1

  # greenwood function
  # https://en.wikipedia.org/wiki/Greenwood_function
  cf[i] = 165.4*(10**(2.1*(i+1.0)/NUM_IHCS)-0.88)
  cf_rad[i] = 2*np.pi*cf[i]

  # must compensate for warping
  # https://en.wikipedia.org/wiki/Bilinear_transform#Frequency_warping
  cf_warp[i] = (2.0*SAMPLE_RATE) * np.arctan((0.5/SAMPLE_RATE)*cf_rad[i])

  # use 4rd order DAPGF and convert to digital filter
  cont_filter_bank[i] = fourth_dapgf(gain[i], cf_rad[i])

  ## method 1: cont2discrete
  filter_bank[i] = signal.cont2discrete(first_apgf(gain[i], cf_rad[i]), dt=1.0/SAMPLE_RATE, method="bilinear")
  # don't know why cont2discrete outputs first term as a nested list
  filter_bank[i] = (filter_bank[i][0][0], filter_bank[i][1])
  # generate bank of three LP biquads (aka 3rd order APGF)
  filter_bank[i] = np.tile(np.concatenate(filter_bank[i]), (4, 1))
  # and add on a BP biquad to make a fourth order DAPGF
  filter_bank[i][3] = bp_biquad(10.0, cf_warp[i], SAMPLE_RATE)

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
sys.stderr.write("first pass: whole waveform...\n")

if SHOW_PROGRESS:
  plt.ion()

# process waveform in chunks of a given size
s_time = time.time()

output_bank = dict(zip(range(*SIM_IHCS), [np.zeros(len(s_in)) for _ in range(*SIM_IHCS)]))

# use multiprocessing to speed this up
def _filt_sig(q_in, q_out):
  while True:
    i, coeffs, data, zi = q_in.get(True)
    s_out, zf = signal.sosfilt(coeffs, x=data, zi=zi)
    q_out.put((i, s_out, zf))

mp_q_in = Queue()
mp_q_out = Queue()
mp_p = Pool(NUM_PROCS, _filt_sig, (mp_q_in, mp_q_out))

for s_idx in range(0, len(s_in), CHUNK_SIZE):
  # padding the waveform allows filter to settle
  chunk_in = s_in[s_idx:s_idx+CHUNK_SIZE]
  chunk_in = np.concatenate((chunk_in[:PAD_SIZE], chunk_in[:PAD_SIZE][::-1], chunk_in))

  for i in range(*SIM_IHCS):
    mp_q_in.put((i, filter_bank[i], chunk_in, zi[i]))

  for _ in range(*SIM_IHCS):
    i, s_out, zf = mp_q_out.get(True)
    zi[i] = zf
    output_bank[i][s_idx:s_idx+CHUNK_SIZE] = s_out[2*PAD_SIZE:]

    if SHOW_PROGRESS and i % 100 == 0:
      plt.clf()
      plt.title("IHC CF: %.02f" % cf[i])
      plt.xlabel("Time (sample)")
      plt.ylabel("Amplitude (a.u.)")
      plt.ylim((-1, 1))
      plt.plot(chunk_in, linewidth=1)
      plt.plot(np.clip(s_out, -1, 1), linewidth=1)
      plt.pause(0.001)

    sys.stderr.write("\rchunk: %s ihc: %s cf: %.02f" % (s_idx, i+1, cf[i]))

sys.stderr.write("\ntook %.02f seconds\n" % (time.time() - s_time))

mp_p.terminate()

if SHOW_PROGRESS:
  plt.ioff()

# determine when neurons are activated and adjust gain
sys.stderr.write("second pass: gain feedback loop...\n")
s_time = time.time()

coding_x = []
coding_y = []

timeout = np.zeros(NUM_IHCS, dtype=np.int64)
time_since_last = np.zeros(NUM_IHCS, dtype=np.int64)
rate_estimate = dict(zip(range(*SIM_IHCS), [0 for _ in range(*SIM_IHCS)]))
firings = dict(zip(range(*SIM_IHCS), [[0] for _ in range(*SIM_IHCS)]))

_timeout_reset = int(0.01 * SAMPLE_RATE)

for t in range(1, len(s_in)):
  if t % 100 == 0:
    sys.stderr.write("\rtime %d of %d" % (t, len(s_in)))

  for i in map(int, np.intersect1d(np.where(np.random.poisson(0.01, NUM_IHCS)), range(*SIM_IHCS))):
    if timeout[i] == 0 and \
       output_bank[i][t] > 1.0 and \
       output_bank[i][t-1] < output_bank[i][t]:
      coding_y.append(cf[i])
      coding_x.append(t)

      if time_since_last[i] > 2**12:
        firings[i].append(0)
      else:
        firings[i].append(time_since_last[i])
        time_since_last[i] = 0

      timeout[i] = _timeout_reset

  timeout = np.subtract(timeout, 1).clip(0)
  time_since_last = np.add(time_since_last, 1)

sys.stderr.write("\ntook %.02f seconds\n" % (time.time() - s_time,))

if SHOW_RESULT:
  plt.figure()
  plt.title("Cochlear Encoding")
  plt.xlabel("Time (sample)")
  #plt.ylabel("IHC CF")
  #plt.scatter(coding_x, coding_y, s=2)
  plt.ylabel("log2(IHC CF)")
  plt.scatter(coding_x, np.log2(coding_y), s=2)
  plt.show()

with open("sparse.txt", "wb") as fp:
  for i, dists in firings.items():
    fp.write(bytes("%s\t12\t" % round(cf[i], 2), "utf-8"))

    for d in dists:
      fp.write(int(d).to_bytes(12, "little"))

    fp.write(bytes("\n", "utf-8"))

with open("output.txt", "w") as fp:
  fp.write("".join(["%s\t%.02f\n" % (x, y) for x, y in zip(coding_x, coding_y)]))
