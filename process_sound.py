import sys
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import time
#from impinvar import impinvar
#from scipy.fftpack import fft

SHOW_FILTERS = False
SHOW_PROGRESS = False
SHOW_RESULT = True

POIS_LAMBDA = 0.01
SAMPLE_RATE = 44100.0
NUM_IHCS = 3500
SIM_IHCS = (0, 3500)

# read in signal
s_in = wavfile.read("andy.wav")[1][:,1] / np.iinfo(np.int32).max
#s_in = wavfile.read("wkwttg.wav")[1][:,1] / np.iinfo(np.int32).max
#s_in = s_in[:10000]

def calc_coeffs(Q, w):
  # DAPGF filter, continuous time
  w = np.float64(w)
  Q = np.float64(Q)

  term1 = w / Q
  term2 = w

#  # third order
#  b = np.array((np.power(w, 5.0), 0.0), dtype=np.float64)
#
#  a = np.array((1.0,
#                3*term1,
#                3*np.power(term1, 2) + 3*np.power(term2, 2),
#                np.power(term1, 3) + 6*term1*np.power(term2, 2),
#                3*np.power(term1, 2)*np.power(term2, 2) + 3*np.power(term2, 4),
#                3*term1*np.power(term2, 4),
#                np.power(term2, 6)),
#               dtype=np.float64)

  # first order
  b = np.array((w, 0.0), dtype=np.float64)

  a = np.array((1.0, term1, np.power(term2, 2)), dtype=np.float64)

  return (b, a)

def digital_bode_plot(b, a):
  plt.figure()
  w, h = signal.freqz(b, a, worN=1500)
  plt.subplot(2, 1, 1)
  db = 20*np.log10(np.abs(h))
  plt.plot(w/np.pi, db)
  plt.subplot(2, 1, 2)
  plt.plot(w/np.pi, np.angle(h))
  plt.show()

def sos_bode_plot(sos):
  plt.figure()
  w, h = signal.sosfreqz(sos, worN=1500)
  plt.subplot(2, 1, 1)
  db = 20*np.log10(np.abs(h))
  plt.plot(w/np.pi, db)
  plt.subplot(2, 1, 2)
  plt.plot(w/np.pi, np.angle(h))
  plt.show()

def analog_bode_plot(sys):
  plt.figure()
  x, y, z = signal.bode(sys, w=2*np.pi*np.linspace(20, 22000, num=1000))
  plt.subplot(2, 1, 1)
  plt.plot(x/(2*np.pi), y)
  plt.subplot(2, 1, 2)
  plt.plot(x/(2*np.pi), z)
  plt.show()

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
  gain[i] = 10

  # greenwood function
  # https://en.wikipedia.org/wiki/Greenwood_function
  cf[i] = 165.4*(10**(2.1*(i+1.0)/NUM_IHCS)-0.88)
  cf_rad[i] = 2*np.pi*cf[i]

  # must compensate for warping
  # https://en.wikipedia.org/wiki/Bilinear_transform#Frequency_warping
  cf_warp[i] = (2.0*SAMPLE_RATE) * np.arctan((0.5/SAMPLE_RATE)*cf_rad[i])

  # use 3rd order DAPGF and convert to digital filter
  cont_filter_bank[i] = calc_coeffs(gain[i], cf_rad[i])

  ## method 1: cont2discrete
  filter_bank[i] = signal.cont2discrete(cont_filter_bank[i], dt=1.0/SAMPLE_RATE, method="bilinear")
  # don't know why cont2discrete outputs first term as a nested list
  filter_bank[i] = (filter_bank[i][0][0], filter_bank[i][1])

  # setup initial conditions for lfilter
  zi[i] = signal.lfilter_zi(*filter_bank[i])

  if SHOW_FILTERS:
    plt.clf()

    # frequency response of continuous time filter
    w, mag, phase = signal.bode(cont_filter_bank[i], w=2*np.pi*np.linspace(20, 22000, num=1000))

    plt.subplot(3, 1, 1)
    plt.title("Freq. Resp. Analog CF=%.02f" % cf[i])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("dB")
    plt.plot(w/(2*np.pi), mag)

    # frequency response of discrete time filter
    w, h = signal.freqz(*filter_bank[i], worN=1500)

    plt.subplot(3, 1, 2)
    plt.title("Freq. Resp. Digital CF=%.02f" % cf[i])
    plt.xlabel("Normalized Frequency (rad/sample)")
    plt.ylabel("dB")
    plt.plot(w/np.pi, 20*np.log10(np.abs(h)))

    # plot of testing discrete time filter
    signal_in = np.sin(2*np.pi*cf[i]*np.linspace(0, 0.1, SAMPLE_RATE*0.1)) + 0.5*np.random.randn(int(SAMPLE_RATE*0.1))
    signal_out = signal.filtfilt(*filter_bank[i], x=signal_in)

    plt.subplot(3, 1, 3)
    plt.title("Filter Test CF=%.02f" % cf[i])
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.plot(signal_in)
    plt.plot(signal_out)

    plt.pause(0.01)

  sys.stderr.write("\rihc: %s cf: %.02f cf_rad: %.02f cf_warp: %.02f" %
                   (i+1, cf[i], cf_rad[i], cf_warp[i]))

if SHOW_FILTERS:
  plt.ioff()

sys.stderr.write("\ntook %.02f seconds\n" % (time.time() - s_time,))

# filter using filterbank
sys.stderr.write("processing waveform...\n")

if SHOW_PROGRESS:
  plt.ion()

while True:
  s_time = time.time()

  output_bank = {}

  for i in range(*SIM_IHCS):
    s_out = signal.filtfilt(*filter_bank[i], x=s_in)
    output_bank[i] = s_out

    if SHOW_PROGRESS:
      plt.clf()
      plt.title("IHC CF: %.02f" % cf[i])
      plt.xlabel("Time (sample)")
      plt.ylabel("Amplitude (a.u.)")
      plt.ylim((-1, 1))
      plt.plot(s_in, linewidth=1)
      plt.plot(np.clip(s_out, -1, 1), linewidth=1)
      plt.pause(0.001)

    sys.stderr.write("\rihc: %s cf: %.02f" % (i+1, cf[i]))

  sys.stderr.write("\ntook %.02f seconds\n" % (time.time() - s_time))

  # perform only one iteration
  break

if SHOW_PROGRESS:
  plt.ioff()

# determine when neurons are activated
sys.stderr.write("generating neuronal encoding...\n")
s_time = time.time()

coding_x = []
coding_y = []

timeout = np.zeros(NUM_IHCS, dtype=np.int64)
time_since_last = np.zeros(NUM_IHCS, dtype=np.int64)
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
