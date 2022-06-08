#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
###############################
# cuDF doxygen warnings check #
###############################

# Run doxygen, ignore missing tag files error
TAG_ERROR1="error: Tag file '.*.tag' does not exist or is not a file. Skipping it..."
TAG_ERROR2="error: cannot open tag file .*.tag for writing"
DOXYGEN_STDERR=`cd cpp/doxygen && { cat Doxyfile ; echo QUIET = YES; echo GENERATE_HTML = NO; }  | doxygen - 2>&1 | sed "/\($TAG_ERROR1\|$TAG_ERROR2\)/d"`
RETVAL=$?

if [ "$RETVAL" != "0" ] || [ ! -z "$DOXYGEN_STDERR" ]; then
  echo -e "$DOXYGEN_STDERR"
  RETVAL=1 #because return value is not generated by doxygen 1.8.20
fi

exit $RETVAL
