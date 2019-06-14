#!/usr/bin/env bash


#bash process_nobpe.sh  vocab_size data_dir lang1 lang2

CODES=$1
VOCAB_SIZE=$1     # number of BPE codes
DATA_PATH=$2
lang1=$3
lang2=$4

N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=20      # number of fastText epochs

TOOLS_PATH=$PWD/tools
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext
FULL_VOCAB=$DATA_PATH/full_vocab


SRC_VOCAB=$DATA_PATH/vocab.$lang1
TGT_VOCAB=$DATA_PATH/vocab.$lang2
SRC_BPE=$DATA_PATH/bpe_codes.$lang1
TGT_BPE=$DATA_PATH/bpe_codes.$lang2

SRC_TRAIN=$DATA_PATH/train.$lang1
TGT_TRAIN=$DATA_PATH/train.$lang2
SRC_VALID=$DATA_PATH/valid.$lang1
TGT_VALID=$DATA_PATH/valid.$lang2
SRC_TEST=$DATA_PATH/test.$lang1
TGT_TEST=$DATA_PATH/test.$lang2

PARA_SRC_TRAIN=$DATA_PATH/train.$lang1-$lang2.$lang1.pth
PARA_SRC_VALID=$DATA_PATH/valid.$lang1-$lang2.$lang1.pth
PARA_SRC_TEST=$DATA_PATH/test.$lang1-$lang2.$lang1.pth

PARA_TGT_TRAIN=$DATA_PATH/train.$lang1-$lang2.$lang2.pth
PARA_TGT_VALID=$DATA_PATH/valid.$lang1-$lang2.$lang2.pth
PARA_TGT_TEST=$DATA_PATH/test.$lang1-$lang2.$lang2.pth



#
#
#
#
#
#cat $SRC_TEST | $NORM_PUNC -l $lang1 | $REM_NON_PRINT_CHAR | $TOKENIZER -l $lang1 -no-escape -threads $N_THREADS > $SRC_TEST.tok
#cat $TGT_TEST | $NORM_PUNC -l $lang2 | $REM_NON_PRINT_CHAR | $TOKENIZER -l $lang2 -no-escape -threads $N_THREADS > $TGT_TEST.tok
###
#cat $SRC_VALID | $NORM_PUNC -l $lang1 | $REM_NON_PRINT_CHAR | $TOKENIZER -l $lang1 -no-escape -threads $N_THREADS > $SRC_VALID.tok
#cat $TGT_VALID | $NORM_PUNC -l $lang2 | $REM_NON_PRINT_CHAR | $TOKENIZER -l $lang2 -no-escape -threads $N_THREADS > $TGT_VALID.tok
##
#cat $SRC_TRAIN | $NORM_PUNC -l $lang1 | $REM_NON_PRINT_CHAR | $TOKENIZER -l $lang1 -no-escape -threads $N_THREADS > $SRC_TRAIN.tok
#cat $TGT_TRAIN | $NORM_PUNC -l $lang2 | $REM_NON_PRINT_CHAR | $TOKENIZER -l $lang2 -no-escape -threads $N_THREADS > $TGT_TRAIN.tok
##
#
##
#
##$FASTBPE learnbpe $CODES  $SRC_TRAIN.tok > $SRC_BPE
#$FASTBPE learnbpe $CODES  $TGT_TRAIN.tok > $TGT_BPE
#
#
#
$FASTBPE applybpe $SRC_VALID.tok.$CODES $SRC_VALID.tok $SRC_BPE
$FASTBPE applybpe $TGT_VALID.tok.$CODES $TGT_VALID.tok $TGT_BPE
#
$FASTBPE applybpe $SRC_TEST.tok.$CODES $SRC_TEST.tok $SRC_BPE
$FASTBPE applybpe $TGT_TEST.tok.$CODES $TGT_TEST.tok $TGT_BPE
#
#
$FASTBPE applybpe $SRC_TRAIN.tok.$CODES $SRC_TRAIN.tok $SRC_BPE
$FASTBPE applybpe $TGT_TRAIN.tok.$CODES $TGT_TRAIN.tok $TGT_BPE
#
#
$FASTBPE getvocab $SRC_TRAIN.tok.$CODES  > $SRC_VOCAB
$FASTBPE getvocab $TGT_TRAIN.tok.$CODES  > $TGT_VOCAB



python preprocess.py $SRC_VOCAB  $SRC_VALID.tok.$CODES
python preprocess.py $TGT_VOCAB  $TGT_VALID.tok.$CODES


python preprocess.py $SRC_VOCAB  $SRC_TEST.tok.$CODES
python preprocess.py $TGT_VOCAB  $TGT_TEST.tok.$CODES


#
mv $SRC_TEST.tok.$CODES.pth $SRC_TEST.pth
mv $TGT_TEST.tok.$CODES.pth $TGT_TEST.pth


mv $SRC_VALID.tok.$CODES.pth $SRC_VALID.pth
mv $TGT_VALID.tok.$CODES.pth $TGT_VALID.pth



echo "-----------------------binarize data--------------------"
python preprocess.py $SRC_VOCAB  $SRC_TRAIN.tok.$CODES
python preprocess.py $TGT_VOCAB  $TGT_TRAIN.tok.$CODES
mv $SRC_TRAIN.tok.$CODES.pth $SRC_TRAIN.pth
mv $TGT_TRAIN.tok.$CODES.pth $TGT_TRAIN.pth


cp $SRC_TRAIN.pth $PARA_SRC_TRAIN
cp $SRC_VALID.pth $PARA_SRC_VALID
cp $SRC_TEST.pth  $PARA_SRC_TEST


cp $TGT_TRAIN.pth $PARA_TGT_TRAIN
cp $TGT_VALID.pth $PARA_TGT_VALID
cp $TGT_TEST.pth  $PARA_TGT_TEST


#cat $SRC_TRAIN.tok.$CODES $TGT_TRAIN.tok.$CODES |shuf > $CONCAT_TOK

#$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_TOK -output $CONCAT_TOK

