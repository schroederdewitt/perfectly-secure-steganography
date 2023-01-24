#!/bin/bash
rm -rf datasets/
mkdir datasets/
cd  datasets/
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
rm -rf wikitext-103-v1.zip