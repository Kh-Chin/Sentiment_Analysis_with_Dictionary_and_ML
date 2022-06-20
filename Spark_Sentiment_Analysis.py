from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import udf
from pyspark.ml.feature import Word2Vec, StopWordsRemover
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import jieba
import findspark

findspark.init()
sc = SparkSession.builder.getOrCreate()
directory = "hdfs://master:9000/data/"
df1 = sc.read.format("csv").option("header", "true").load(directory + "waimai_10k.csv")
df2 = sc.read.format("csv").option("header", "true").load(directory + "ChnSentiCorp_htl_all.csv")
df3 = sc.read.format("csv").option("header", "true").load(directory + "online_shopping_10_cats.csv")
df = df1.union(df2).union(df3.drop("cat"))


def cut(x):
    if x:
        s = jieba.lcut(x)
        return s
    return ["EMPTY"]


cut_udf = udf(cut, ArrayType(StringType()))
df = df.withColumn("cut", cut_udf(df.review))
df = df.withColumn("label", df.label.cast("int"))

stopwords = sc.read.format("text").option("header", "true").load(directory + "stoplist1.txt")
remover = StopWordsRemover(inputCol="cut", outputCol="stop", stopWords=[row["value"] for row in stopwords.collect()])
df = remover.transform(df)

train, test = df.randomSplit([0.8, 0.2])
w2v = Word2Vec(vectorSize=100, minCount=3, numPartitions=6, inputCol="stop", outputCol="vec")
w2v_model = w2v.fit(train)
train = w2v_model.transform(train)
test = w2v_model.transform(test)

train.show()
test.show()

SVM = LinearSVC(featuresCol="vec", labelCol="label")
SVM_model = SVM.fit(train)

LR = LogisticRegression(featuresCol="vec", labelCol="label")
LR_model = LR.fit(train)

SVM_predictions = SVM_model.transform(test)
LR_predictions = LR_model.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
print("SVM model accuracy:" + str(evaluator.evaluate(SVM_predictions)))
print("Logistic Regression model accuracy:" + str(evaluator.evaluate(LR_predictions)))
