import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv("UberDataset.csv")
dataset.head()
print(dataset)
print(dataset.shape)
print(dataset.info())
dataset["PURPOSE"].fillna("NOT", inplace=True)
print(dataset)
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'],
                                       errors='coerce')
dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'],
                                     errors='coerce')
print(dataset)
from datetime import datetime

# Splitting the START_DATE to date and time column and then converting the time into four different categories i.e. Morning, Afternoon, Evening, Night

dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour

#changing into categories of day and night
dataset['day-night'] = pd.cut(x=dataset['time'],
                              bins = [0,10,15,19,24],
                              labels = ['Morning','Afternoon','Evening','Night'])
print(dataset)
dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)
print(dataset)

# checking the unique value in dataset of the column with object data type

obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

unique_values = {}
for col in object_cols:
  unique_values[col] = dataset[col].unique().size
print(unique_values)

# Now, we will be using matplotlib and seaborn library for countplot the CATEGORY and PURPOSE columns.

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.countplot(dataset['CATEGORY'])
plt.xticks(rotation=90)

plt.subplot(1,2,2)
sns.countplot(dataset['PURPOSE'])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Let’s do the same for time column, here we will be using the time column which we have extracted above.
sns.countplot(dataset['day-night'])
plt.xticks(rotation=90)
plt.show()

# Now, we will be comparing the two different categories along with the PURPOSE of the user.
plt.figure(figsize=(15, 5))
sns.countplot(data=dataset, x='PURPOSE', hue='CATEGORY')
plt.xticks(rotation=90)
plt.show()



# After that, we can now find the correlation between the columns using heatmap.
# Select only numerical columns for correlation calculation
numeric_dataset = dataset.select_dtypes(include=['number'])

sns.heatmap(numeric_dataset.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)

# This code is modified by Susobhan Akhuli

dataset['MONTH'] = pd.DatetimeIndex(dataset['START_DATE']).month
month_label = {1.0: 'Jan', 2.0: 'Feb', 3.0: 'Mar', 4.0: 'April',
               5.0: 'May', 6.0: 'June', 7.0: 'July', 8.0: 'Aug',
               9.0: 'Sep', 10.0: 'Oct', 11.0: 'Nov', 12.0: 'Dec'}
dataset["MONTH"] = dataset.MONTH.map(month_label)

mon = dataset.MONTH.value_counts(sort=False)

# Month total rides count vs Month ride max count
df = pd.DataFrame({"MONTHS": mon.values,
                   "VALUE COUNT": dataset.groupby('MONTH',
                                                  sort=False)['MILES'].max()})

p = sns.lineplot(data=df)
p.set(xlabel="MONTHS", ylabel="VALUE COUNT")
plt.show()

dataset['DAY'] = dataset.START_DATE.dt.weekday
day_label = {
    0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thus', 4: 'Fri', 5: 'Sat', 6: 'Sun'
}
dataset['DAY'] = dataset['DAY'].map(day_label)
day_label = dataset.DAY.value_counts()
sns.barplot(x=day_label.index, y=day_label)
plt.xlabel('DAY')
plt.ylabel('COUNT')
plt.show()

# Now, let’s explore the MILES Column .

# We can use boxplot to check the distribution of the column.

sns.boxplot(dataset['MILES'])
plt.show()

# As the graph is not clearly understandable. Let’s zoom in it for values lees than 100.

sns.boxplot(dataset[dataset['MILES']<100]['MILES'])
plt.show()

# It’s bit visible. But to get more clarity we can use distplot for values less than 40.

sns.distplot(dataset[dataset['MILES']<40]['MILES'])
plt.show()






