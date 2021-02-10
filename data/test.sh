#This script invokes Multiwfn to qtaim variables

#!/bin/bash
icc=0
nfile=`ls ../xyz/DB3/*.xyz|wc -l`
for inf in ../xyz/DB3/*.xyz
do
((icc++))
echo Running ${inf} ... \($icc of $nfile\)
./nwchem ${inf} << EOF > tmp.txt
EOF
#new="${inf}.sum"
#mv CPprop.txt ${new}
echo ${inf} has finished
echo
done
