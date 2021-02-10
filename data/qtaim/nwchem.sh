#!/bin/bash
icc=0
nfile=`ls *.in | wc -l`
for inf in *.in
do

((icc++))

echo Running ${inf} ... \($icc of $nfile\)
nwchem ${inf}
if [[$icc % 10]]
then
rm *b
rm *p
rm *b^-1
rm *c
rm *cfock
rm *db
rm *zmat
rm *movecs
rm *oexch
fi

rm ${inf}
echo ${inf} has finished
echo DONE

done
