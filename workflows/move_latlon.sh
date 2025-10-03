#!/bin/bash

set -e

echo "Script to move lat-lon"

InputFile=$1
OutputFile=$2

TempFile=${InputFile}.1.nc
rm -f $TempFile

echo "Reverse lat"
ncap2 -s 'lat=-lat' $InputFile $TempFile

InputFile=$TempFile
TempFile=${InputFile}.2.nc
rm -f $TempFile

echo "Move lon"
ncks -O --msa -d lon,180.,360. -d lon,0.,180.0 $InputFile $TempFile
rm -f $InputFile

echo "Update lon"
ncap2 -s 'where(lon > 180) lon=lon-360' $TempFile $OutputFile
rm -f $TempFile

exit 0

