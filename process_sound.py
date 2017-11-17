import random
import sys
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import time
from impinvar import impinvar

SHOW_FILTERS = True

FLOAT_MIN = 1e-10
SAMPLE_RATE = 44100.0
NUM_IHCS = 3500
SIM_IHCS = (0, 3500)

# generate signal
num_seconds = 1.0
test_freq = 207.12

#xvals = np.linspace(0, num_seconds, SAMPLE_RATE*num_seconds)
#s_in = np.sin(2 * np.pi * test_freq * xvals)
#s_in = np.sin(2 * np.pi * test_freq * xvals) + np.sin(2 * np.pi * 213.12 * xvals) + 0.1*np.random.rand(int(SAMPLE_RATE*num_seconds))
s_in = wavfile.read("andy.wav")[1][:,1] / np.iinfo(np.int32).max

def calc_coeffs(Q, w):
  # DAPGF filter, continuous time
  term1 = w / float(Q)
  term2 = float(w)

  # third order
  b = [w**5.0, 0.0]

  a = [1.0,
       3*term1,
       (3*term1**2 + 3*term2**2),
       (term1**3 + 6*term1*term2**2),
       (3*term1**2*term2**2 + 3*term2**4),
       3*term1*term2**4,
       term2**6]

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

plt.ion()
plt.subplots_adjust(hspace=1.0)

for i in range(*SIM_IHCS):
  # start with gain (q) of 1/2**0.5 (minimum for 0 dB gain)
  #gain[i] = 1/2**0.5
  gain[i] = 5.0

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
    signal_out = signal.lfilter(*filter_bank[i], signal_in)

    plt.subplot(3, 1, 3)
    plt.title("Filter Test CF=%.02f" % cf[i])
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.plot(signal_in)
    plt.plot(signal_out)

    plt.pause(0.01)

  sys.stderr.write("\rihc: %s cf: %.02f cf_rad: %.04f cf_warp: %s" %
                   (i+1, cf[i], cf_rad[i], cf_warp[i]))

plt.ioff()

sys.stderr.write("\n")

# filter using filterbank
plt.ion()

while True:
  s_time = time.time()

  test_freq_cf = []
  test_freq_output = []
  pps = []
  gains = []
  output_bank = {}

  for i in range(*SIM_IHCS):
    s_out, zf = signal.lfilter(*filter_bank[i], x=s_in, zi=zi[i])
    zi[i] = zf

    output_bank[i] = s_out

    plt.clf()
    plt.title("IHC CF: %.02f" % cf[i])
    plt.xlabel("Time (sample)")
    plt.ylabel("Amplitude (a.u.)")
    plt.plot(s_in, linewidth=1)
    plt.plot(s_out, linewidth=1)
    plt.pause(0.001)

#    freqz = fft(s_out, int(SAMPLE_RATE))

#    gains.append(gain[i])
#    test_freq_cf.append(cf[i])
#    test_freq_output.append(abs(freqz[int(cf[i])]))

    ## SHOW FFT
    #plt.figure()
    #plt.plot(abs(freqz))
    #plt.show()

#    look_back = int(2 * SAMPLE_RATE / 100.0)
#    peak_width = 0.5 * SAMPLE_RATE / 100.0

#    peaks_per_sec = 0#(s_out[-look_back:][signal.find_peaks_cwt(s_out[-look_back:], [1])] > 1.0).sum() * (SAMPLE_RATE / look_back)
#    pps.append(peaks_per_sec)
#
#    # adjust gain based on maximum of 100 firings per second
#    if peaks_per_sec == 0.0 and gain[i] == 0.0:
#      gain[i] = 0.1
#    elif peaks_per_sec > 100.0:
#      gain[i] *= np.exp(-0.1*(peaks_per_sec/100.0))
#    elif peaks_per_sec < 10.0:
#      gain[i] *= np.exp(0.001*(100.0/(peaks_per_sec+1)))
#
#    if gain[i] < 0.0:
#      gain[i] = 0.0
#    elif gain[i] > 100.0:
#      gain[i] = 100.0
#
#    cont_filter_bank[i] = calc_coeffs(gain[i], cf_rad[i])
#
#    ## method 1: cont2discrete
#    filter_bank[i] = signal.cont2discrete(cont_filter_bank[i], dt=1.0/SAMPLE_RATE, method="bilinear")
#    # don't know why cont2discrete outputs first term as a nested list
#    filter_bank[i] = (filter_bank[i][0][0], filter_bank[i][1])
#
#    zi[i] = signal.lfilter_zi(*filter_bank[i])

    sys.stderr.write("\rihc: %s cf: %.02f gain: %.02f" % (i, cf[i], gain[i]))
#    print("ihc: %s cf: %.02f gain: %.02f max_out: %.02f peaks_per_sec: %d" % (i, cf[i], gain[i], max(s_out), peaks_per_sec))

#  plt.clf()
#  plt.subplot(3, 1, 1)
#  plt.semilogy(test_freq_cf, test_freq_output)
#  plt.subplot(3, 1, 2)
#  plt.plot(test_freq_cf, pps)
#  plt.subplot(3, 1, 3)
#  plt.plot(test_freq_cf, gains)
#  plt.pause(0.05)

  print("\ntook %.02f seconds\n\n" % (time.time() - s_time))

  break

plt.ioff()

# determine when neurons are activated
coding_x = []
coding_y = []

timeout = dict(zip(range(*SIM_IHCS), [0] * len(range(*SIM_IHCS))))
time_since_last = dict(zip(range(*SIM_IHCS), [0] * len(range(*SIM_IHCS))))
firings = dict(zip(range(*SIM_IHCS), [[0] for _ in range(*SIM_IHCS)]))

for t in range(1, len(s_in)):
  for i in range(*SIM_IHCS):
    #print(i, t, output_bank[i][t], np.random.poisson(lam=0.1), timeout[i])
    if output_bank[i][t] > 1.0 and \
       output_bank[i][t-1] < output_bank[i][t] and \
       np.random.poisson(lam=0.01) > 0 and \
       timeout[i] == 0:
      coding_y.append(cf[i])
      coding_x.append(t)

      if time_since_last[i] > 2**12:
        firings[i].append(0)
      else:
        firings[i].append(time_since_last[i])
        time_since_last[i] = 0

      timeout[i] = int(0.01 * SAMPLE_RATE)

    if timeout[i] > 0:
      timeout[i] = timeout[i] - 1

    time_since_last[i] += 1

plt.figure()
plt.title("Cochlear Encoding")
plt.xlabel("Time (sample)")
plt.ylabel("IHC CF")
#plt.plot(range(1, len(s_in)), output_bank[533][1:])
#plt.scatter(coding_x, [0] * len(coding_x))
plt.scatter(coding_x, coding_y, s=2)
plt.show()

with open("sparse.txt", "wb") as fp:
  for i, dists in firings.items():
    fp.write(bytes("%s\t12\t" % round(cf[i], 2), "utf-8"))

    for d in dists:
      fp.write(int(d).to_bytes(12, "little"))

    fp.write(bytes("\n", "utf-8"))

with open("output.txt", "w") as fp:
  fp.write("".join(["%s\t%.02f\n" % (x, y) for x, y in zip(coding_x, coding_y)]))
