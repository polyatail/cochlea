a <- read.table("output.txt")

### guess for entire volley of IHCs
#
#ndist_diff = c()
#for (i in 2:length(unique(a$V1))) {
#  ndist_diff = c(ndist_diff, unique(a$V1)[i] - unique(a$V1)[i-1])
#}
#guesses = c()
#for (guess in seq(1, 220.5*8, 0.1)) { guesses = c(guesses, sum(abs(round(ndist_diff/guess) - (ndist_diff/guess)))) }
#res = cbind(seq(1, 220.5*8, 0.1), guesses)
#print(ndist_diff)
#print(head(res[order(res[,2]),], n=5))

## guess for single IHCs

for (i in unique(as.numeric(a$V2)))
{
  this_ihc = a[which(as.numeric(a$V2) == i),]

  ndist_diff = c()

  for (j in 2:length(this_ihc$V1)) {
    ndist_diff = c(ndist_diff, this_ihc$V1[j] - this_ihc$V1[j-1])
  }

  guess_samps = seq(44100/(this_ihc$V2[1]+75), 44100/(this_ihc$V2[1]-75), 0.1)
  guess_freqs = 44100/guess_samps
  guess_freq_diffs = guess_freqs - this_ihc$V2[1]

  guesses = c()

  for (guess in guess_samps) {
    guesses = c(guesses, sum(abs(round(ndist_diff/guess) - (ndist_diff/guess))))
  }

  res = cbind(guess_freqs, guess_freq_diffs, guesses)
  print(sprintf("### GUESS FOR IHC CF=%.02f", this_ihc$V2[1]))
  print(head(res[order(res[,3]),], n=5))
}
