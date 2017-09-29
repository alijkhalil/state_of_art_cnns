#!/bin/sh

# Set up variables and local directory
UTIL_REPO_NAME="dl_utilities"
UTIL_REPO_GIT_HASH="8e4065c1a32c55110c9b48cca7b6108408ea85b7"

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
