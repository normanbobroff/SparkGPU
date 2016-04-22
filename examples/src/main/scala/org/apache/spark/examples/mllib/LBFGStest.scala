/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples.mllib

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.optimization.LBFGS
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.optimization.NativeLBFGS

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.StreamingLogisticRegressionWithSGD
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}

object LBFGStest {



  def isHeader(line: String): Boolean = {
     line.contains("label")
  }


  def parse(line: String) = {
    val pieces = line.split("\t")
    val label = pieces(0).toDouble
    // val features = pieces.slice(1, 24)
    val features = pieces.tail
    val scores = new DenseVector(features.map(s => s.toDouble))
    new LabeledPoint(label, scores)
  }

  private def sampleRatio(train: RDD[LabeledPoint]) = {
    train.filter{case(lp) => lp.label > 0.5}.count().doubleValue()/train.count()
  }


  def main(args: Array[String]) {

    val t0 = System.nanoTime()

    if (args.length != 1) {
      System.err.println(
        "Usage: LBFGStest <file.tsv>")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName(s"LBFGS possibly with GPU")
    val sc = new SparkContext(conf)

    val rawData = sc.textFile(args(0))
    val valueRows = rawData.filter(x => !isHeader(x))
    val train_data = valueRows.map(line => parse(line))
    val initialWeightsWithIntercept = Vectors.dense(new Array[Double](train_data.first.features.size + 1))
    
    val numCorrections: Int = 20
    val convergenceTol: Double = 0.0001    
    val maxNumIterations:Int = 500
    val regParam: Double = 0.1
    val truePositiveFraction: Double = -1.0
    val weightedCostFunctionCorrection : Boolean = false
    
    val t1 = System.nanoTime()
    val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
		train_data.map(x=>(x.label, MLUtils.appendBias(x.features))),
		new LogisticGradient(),
                new SquaredL2Updater(),
                numCorrections,
                convergenceTol,
                maxNumIterations,
                regParam,
                initialWeightsWithIntercept)
    val t2 = System.nanoTime()
    
    

    val weights = Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1))
    val intercept =weightsWithIntercept(weightsWithIntercept.size - 1)
    val minimum = loss(loss.size - 1)
    /*
    println("")
    println("WEIGHTS:")
    weights.toArray.iterator foreach println
    
    println("")
    println("LOSS HISTORY:")
    loss.iterator foreach println
   */
    println("LBFGS FILE          : " + args(0))
    println("SPARK_IMPLEMENTATION: " + System.getenv("SPARK_IMPLEMENTATION"))
    println("MINIMUM             : " + minimum)
    println("INTERCEPT           : " + intercept)
    println("TEST SETUP TIME     : " + (t1 - t0)/1000000000 + " sec")
    println("runLBFGS TIME       : " + (t2 - t1)/1000000000 + " sec")
  }

}
