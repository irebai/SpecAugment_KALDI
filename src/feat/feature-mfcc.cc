// feat/feature-mfcc.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek
//                2016  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "feat/feature-mfcc.h"
#include <time.h>       /* time */


namespace kaldi {


void MfccComputer::Compute(BaseFloat signal_log_energy,
                           BaseFloat vtln_warp,
                           VectorBase<BaseFloat> *signal_frame,
                           VectorBase<BaseFloat> *feature) {
  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());

  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  if (opts_.use_energy && !opts_.raw_energy)
    signal_log_energy = Log(std::max<BaseFloat>(VecVec(*signal_frame, *signal_frame),
                                     std::numeric_limits<float>::min()));

  if (srfft_ != NULL)  // Compute FFT using the split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two.
    RealFft(signal_frame, true);

  // Convert the FFT into a power spectrum.
  ComputePowerSpectrum(signal_frame);
  SubVector<BaseFloat> power_spectrum(*signal_frame, 0,
                                      signal_frame->Dim() / 2 + 1);

  mel_banks.Compute(power_spectrum, &mel_energies_);

  // avoid log of zero (which should be prevented anyway by dithering).
  mel_energies_.ApplyFloor(std::numeric_limits<float>::epsilon());
  mel_energies_.ApplyLog();  // take the log.

  //KALDI_LOG << mel_energies_;

  feature->SetZero();  // in case there were NaNs.
  // feature = dct_matrix_ * mel_energies [which now have log]
  feature->AddMatVec(1.0, dct_matrix_, kNoTrans, mel_energies_, 0.0);

  if (opts_.cepstral_lifter != 0.0)
    feature->MulElements(lifter_coeffs_);

  if (opts_.use_energy) {
    if (opts_.energy_floor > 0.0 && signal_log_energy < log_energy_floor_)
      signal_log_energy = log_energy_floor_;
    (*feature)(0) = signal_log_energy;
  }

  if (opts_.htk_compat) {
    BaseFloat energy = (*feature)(0);
    for (int32 i = 0; i < opts_.num_ceps - 1; i++)
      (*feature)(i) = (*feature)(i+1);
    if (!opts_.use_energy)
      energy *= M_SQRT2;  // scale on C0 (actually removing a scale
    // we previously added that's part of one common definition of
    // the cosine transform.)
    (*feature)(opts_.num_ceps - 1)  = energy;
  }
}



void MfccComputer::ComputeMask(VectorBase<BaseFloat> *signal_log_energy,
                        BaseFloat vtln_warp,
                        int32 F, int32 T, int32 mF, int32 mT,
                        Matrix<BaseFloat> windows,
                        Matrix<BaseFloat> *features) {
  KALDI_ASSERT(windows.NumCols() == opts_.frame_opts.PaddedWindowSize() &&
               features->NumCols() == this->Dim());
  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  //compute power spectrum
  Matrix<BaseFloat> power_spectrums;
  power_spectrums.Resize(features->NumRows(), windows.NumCols() / 2 + 1);
  for(int32 i=0; i < power_spectrums.NumRows(); i++){
    SubVector<BaseFloat> signal_frame(windows, 0);
    if (opts_.use_energy && !opts_.raw_energy)
      (*signal_log_energy)(i) = Log(std::max<BaseFloat>(VecVec(signal_frame, signal_frame),
                                      std::numeric_limits<float>::min()));

    if (srfft_ != NULL)  // Compute FFT using the split-radix algorithm.
      srfft_->Compute(signal_frame.Data(), true);
    else  // An alternative algorithm that works for non-powers-of-two.
      RealFft(&signal_frame, true);

    // Convert the FFT into a power spectrum.
    ComputePowerSpectrum(&signal_frame);
    SubVector<BaseFloat> power_spectrum(signal_frame, 0,
                                        signal_frame.Dim() / 2 + 1);
    power_spectrums.CopyRowFromVec(power_spectrum,i);
    windows.RemoveRow(0);
  }

  //apply frequency and time masks
  /* initialize random seed: */
  srand (time(NULL));
  BaseFloat mean = power_spectrums.Sum() / (power_spectrums.NumCols() * power_spectrums.NumRows());
  //1. frequence mask
  if(F>0){
    int32 mel_freq = power_spectrums.NumCols();
    for(int16 i=0; i<mF; i++){
      int32 f = rand() % F;
      int32 f_zero = rand() % (mel_freq - f);
      if(f_zero != f_zero + f){
        SubMatrix<BaseFloat> freq = power_spectrums.ColRange(f_zero, f);
        freq.Set(mean);
      }
    }
  }
  //2. time mask 
  if(T>0){
    int32 time = power_spectrums.NumRows();
    for(int16 i=0; i<mT; i++){
      int32 t = rand() % T;
      int32 t_zero = rand() % (time - t);
      if(t_zero != t_zero + t){
        SubMatrix<BaseFloat> tm = power_spectrums.RowRange(t_zero, t);
        tm.Set(mean);
      }
    }
  }
  //KALDI_LOG << power_spectrums.Row(features->NumRows()-1);

  features->SetZero();  // in case there were NaNs.
  //KALDI_LOG << features->NumRows() << " " << features->NumCols() << " " << power_spectrums.NumRows();
  for(int32 i=0; i < features->NumRows(); i++){
    SubVector<BaseFloat> power_spectrum(power_spectrums, 0);
    Vector<BaseFloat> feature;
    feature.Resize(features->NumCols());
    //convert power spectrum to cepstral features
    mel_banks.Compute(power_spectrum, &mel_energies_);

    // avoid log of zero (which should be prevented anyway by dithering).
    mel_energies_.ApplyFloor(std::numeric_limits<float>::epsilon());
    mel_energies_.ApplyLog();  // take the log.

    // feature = dct_matrix_ * mel_energies [which now have log]
    feature.AddMatVec(1.0, dct_matrix_, kNoTrans, mel_energies_, 0.0);

    if (opts_.cepstral_lifter != 0.0)
      feature.MulElements(lifter_coeffs_);

    if (opts_.use_energy) {
      if (opts_.energy_floor > 0.0 && (*signal_log_energy)(i) < log_energy_floor_)
        (*signal_log_energy)(i) = log_energy_floor_;
      feature(0) = (*signal_log_energy)(i);
    }
    features->CopyRowFromVec(feature,i);
    power_spectrums.RemoveRow(0);
  }
  //KALDI_LOG << features->NumRows() << " " << features->NumCols();
  //KALDI_LOG << features->Row(0);

/*
    KALDI_LOG << power_spectrum.Dim() << " " << windows.NumCols();
    for(int32 j=0; j < features->NumCols(); j++){
      (*features)(i,j)=1;
    }
*/

}


MfccComputer::MfccComputer(const MfccOptions &opts):
    opts_(opts), srfft_(NULL),
    mel_energies_(opts.mel_opts.num_bins) {

  int32 num_bins = opts.mel_opts.num_bins;
  if (opts.num_ceps > num_bins)
    KALDI_ERR << "num-ceps cannot be larger than num-mel-bins."
              << " It should be smaller or equal. You provided num-ceps: "
              << opts.num_ceps << "  and num-mel-bins: "
              << num_bins;

  Matrix<BaseFloat> dct_matrix(num_bins, num_bins);
  ComputeDctMatrix(&dct_matrix);
  // Note that we include zeroth dct in either case.  If using the
  // energy we replace this with the energy.  This means a different
  // ordering of features than HTK.
  SubMatrix<BaseFloat> dct_rows(dct_matrix, 0, opts.num_ceps, 0, num_bins);
  dct_matrix_.Resize(opts.num_ceps, num_bins);
  dct_matrix_.CopyFromMat(dct_rows);  // subset of rows.
  if (opts.cepstral_lifter != 0.0) {
    lifter_coeffs_.Resize(opts.num_ceps);
    ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);
  }
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = Log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0);
}

MfccComputer::MfccComputer(const MfccComputer &other):
    opts_(other.opts_), lifter_coeffs_(other.lifter_coeffs_),
    dct_matrix_(other.dct_matrix_),
    log_energy_floor_(other.log_energy_floor_),
    mel_banks_(other.mel_banks_),
    srfft_(NULL),
    mel_energies_(other.mel_energies_.Dim(), kUndefined) {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
       iter != mel_banks_.end(); ++iter)
    iter->second = new MelBanks(*(iter->second));
  if (other.srfft_ != NULL)
    srfft_ = new SplitRadixRealFft<BaseFloat>(*(other.srfft_));
}



MfccComputer::~MfccComputer() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    delete iter->second;
  delete srfft_;
}

const MelBanks *MfccComputer::GetMelBanks(BaseFloat vtln_warp) {
  MelBanks *this_mel_banks = NULL;
  std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(opts_.mel_opts,
                                  opts_.frame_opts,
                                  vtln_warp);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}



}  // namespace kaldi
