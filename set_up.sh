#!/bin/sh

# Set up variables and local directory
UTIL_REPO_NAME="dl_utilities"
UTIL_REPO_GIT_HASH="6ed7c4c8e089e28a15c32e1b8dfb071d16e2c6c2"

cd `dirname $0`


# Get necessary repo's
echo -n "Getting other repositories needed for the project...  "

if [ ! -d $UTIL_REPO_NAME ]; then
    git clone git@github.com:alijkhalil/"$UTIL_REPO_NAME".git > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Download error!"
        exit 1
    fi
fi

cd $UTIL_REPO_NAME
git checkout $UTIL_REPO_GIT_HASH > /dev/null 2>&1
cd - > /dev/null 2>&1


# Print success and exit
echo "Done!"
exit 0
