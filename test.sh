#!/usr/bin/env bash
PYSCRIPT=infer-app.py
OUTFILE=out.tsv
INFILE=test_data.shame 
P01=0.1
P10=0.2
PRIOR1=0.4

case $1 in
   run)
       python ${PYSCRIPT} --sham ${INFILE} --out ${OUTFILE} --p01  ${P01} --p10 ${P10}  --prior1 ${PRIOR1}
       echo $! 
   ;;
   clear)
      rm ${OUTFILE} 2>/dev/null
      echo "${OUTFILE} removed"
   ;;
   *)
      echo "usage: pdm_xxxapi {run|clear}" ;;
esac
exit 0