#!/bin/bash

cd ~/scratch

curl -L https://yihui.org/tinytex/install-bin-unix.sh | sh

export PATH="$HOME/bin:$PATH"

tlmgr install type1cm cm-super dvipng latex-extra latex-recommended underscore
