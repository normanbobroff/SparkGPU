#!/bin/bash

// May need to stop zinc
./stopzinc.sh

./build/mvn -DskipTests package
