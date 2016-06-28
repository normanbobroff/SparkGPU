./bin/spark-submit \
  --class "com.ibm.watson.dev.WordCount" \
  --master local[4]  \
  target/normSparkWordCount-0.0.1-SNAPSHOT.jar
