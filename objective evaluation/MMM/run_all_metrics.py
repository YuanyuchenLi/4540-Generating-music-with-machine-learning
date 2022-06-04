import os, sys
import warnings
sys.path.append('./musdr/')
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from glob import glob
from argparse import ArgumentParser
from musdr.eval_metrics import (
  compute_piece_pitch_entropy,
  compute_piece_groove_similarity,
  compute_piece_chord_progression_irregularity,
  compute_structure_indicator
)
from musdr.side_utils import get_event_seq

def write_report(result_dict, out_csv_file):
  df = pd.DataFrame().from_dict(result_dict)
  df = df.append(df.agg(['mean']))
  df = df.round(4)
  df.loc['mean', 'piece_name'] = 'DATASET_MEAN'
  df.to_csv(out_csv_file, index=False, encoding='utf-8')

if __name__ == "__main__":

  test_pieces = sorted( glob(os.path.join('musdr/testdata/tr_csv', '*')) )
  test_pieces_scplot = sorted( glob(os.path.join('musdr/testdata/tr_scp', '*')) )
  print (test_pieces, test_pieces_scplot)
  result_dict = {
    'piece_name': [],
    'H1': [],
    'H4': [],
    'GS': [],
    'CPI': [],
    'SI_short': [],
    'SI_mid': [],
    'SI_long': []    
  }

  assert len(test_pieces) == len(test_pieces_scplot), 'detected discrepancies between 2 input directories.'

  for p, p_sc in zip(test_pieces, test_pieces_scplot):
    print ('>> now processing: {}'.format(p))
    seq = get_event_seq(p)
    result_dict['piece_name'].append(p.replace('\\', '/').split('/')[-1])
    h1 = compute_piece_pitch_entropy(seq, 1)
    result_dict['H1'].append(h1)
    h4 = compute_piece_pitch_entropy(seq, 4)
    result_dict['H4'].append(h4)
    gs = compute_piece_groove_similarity(seq)
    result_dict['GS'].append(gs)
    cpi = compute_piece_chord_progression_irregularity(seq)
    result_dict['CPI'].append(cpi)
    si_short = compute_structure_indicator(p_sc, 3, 8)
    result_dict['SI_short'].append(si_short)
    si_mid = compute_structure_indicator(p_sc, 8, 15)
    result_dict['SI_mid'].append(si_mid)
    si_long = compute_structure_indicator(p_sc, 15)
    result_dict['SI_long'].append(si_long)

    print ('  1-bar H: {:.3f}'.format(h1))
    print ('  4-bar H: {:.3f}'.format(h4))
    print ('  GS: {:.4f}'.format(gs))
    print ('  CPI: {:.4f}'.format(cpi))
    print ('  SI_short: {:.4f}'.format(si_short))
    print ('  SI_mid: {:.4f}'.format(si_mid))
    print ('  SI_long: {:.4f}'.format(si_long))
    print ('==========================')

  if len(result_dict):
    write_report(result_dict, 'musdr/testdata/out')
  else:
    print ('No pieces are found !!')
