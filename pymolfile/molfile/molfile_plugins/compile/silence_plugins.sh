#!/bin/bash

fpath="../molfile_plugin/src/"
#ipath="../include/"
#incfile="silence_plugins.h"

allpattern="printf"
pattern1=" printf("
pattern2=" fprintf(stderr"
pattern3=" fprintf(stdout"
addstr1=" sprintf(msgstr"
addstr2=" sprintf(msgerr"
addstr3=" sprintf(msgout"
adddef1="#ifndef SILENCEMESSAGES "
adddef2="char msgstr[512], msgout[512], msgerr[512];"
adddef3="#endif"
alsomatch=");"
ignore1="/*"
ignore2="//"
exclude="("

files=`find -E ${fpath}* -regex '.*\.(c|h|cxx|hxx|C|H)' -type f -exec grep -l -e "$allpattern" {} \; | grep -v "periodic_table"`

echo "Changing lines with $pattern1 $pattern2 and $pattern3 in following files:"
#echo "Copying $incfile to $ipath"
#cp $incfile $ipath
fcount=1

for file in $files
do
	if [ -f $file.bak ]; then
	    cp $file.bak $file
	fi
	echo "Processing file $file ..."
	#incline=` cat $file | grep -n -e "#" | head -n 1 | awk -F":" '{print $1}'`
	#sed -i.bak $incline'a\'$'\n''#include "silence_plugins.h"'$'\n' $file
	#sed -i.bak $incline'a\'$'\n''#endif'$'\n' $file
	#firstline=`cat $file | head -n 1`
	#sed -i.bak '1a\'$'\n'${firstline}$'\n' $file
	sed -i.bak '1i\'$'\n''#endif'$'\n' $file
	sed -i "" '1i\'$'\n''char msgstr'${fcount}'[512], msgout'${fcount}'[512], msgerr'${fcount}'[512];'$'\n' $file
	sed -i "" '1i\'$'\n''#define SILENCE_'${fcount}'_PLUGIN 1'$'\n' $file 
	sed -i "" '1i\'$'\n''#ifndef SILENCE_'${fcount}'_PLUGIN'$'\n' $file 
	#sed -i.bak $incline'a\'$'\n''#define SILENCE_PLUGINS_'${fcount}' 1'$'\n' $file 
	#sed -i.bak $incline'a\'$'\n''#ifndef SILENCE_PLUGINS_'${fcount}''$'\n' $file 
	singlelines1=`cat $file | grep -n -e "$pattern1" | grep -v "^--" | awk -F":" '{print $1}'`
	singlelines2=`cat $file | grep -n -e "$pattern2" | grep -v "^--" | awk -F":" '{print $1}'`
	singlelines3=`cat $file | grep -n -e "$pattern3" | grep -v "^--" | awk -F":" '{print $1}'`
	for line in $singlelines1
	do
		sed -i "" $line's|'"$pattern1"'|'"${addstr1}${fcount},"'|' $file
	done
	for line in $singlelines2
	do
		sed -i "" $line's|'"$pattern2"'|'"${addstr2}${fcount}"'|' $file
	done
	for line in $singlelines3
	do
		sed -i "" $line's|'"$pattern3"'|'"${addstr3}${fcount}"'|' $file
	done
	#if [ -f $file.bak2 ]; then
	#    rm $file.bak2
        #fi
	let fcount="$fcount + 1"
done

