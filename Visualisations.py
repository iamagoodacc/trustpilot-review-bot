from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
#changing JSON to CSV
json_file_path = r'C:\Users\Nish\Documents\Trustpilot Review App\asda.json'
csv_file_path = r'C:\Users\Nish\Documents\Trustpilot Review App\TrustpilotReviews.csv'  
df = pd.read_json(json_file_path)
df.to_csv(csv_file_path, index=False)

print(f"JSON data has been successfully converted to {csv_file_path}")

# creating dataframe
df = pd.read_csv(r'C:\Users\Nish\Documents\Trustpilot Review App\TrustpilotReviews.csv', index_col=0)

# joining all reviews together
text = " ".join(review for review in df.body)

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["ANONYMIZED", "EMAIL", "DATE", "LOCATION", "PERSON", "Asda"])

# Create and generate a word cloud image:
wordcloud = WordCloud(width=800, height=800,stopwords=stopwords, max_words=500, max_font_size=100, background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# loading the CSV file
data = pd.read_csv(r"C:\Users\Nish\Documents\Trustpilot Review App\TrustpilotReviews.csv")

sns.scatterplot(data=data, x="timestamp", y="rating")
plt.show()

# printing out useful frequency plots
for column in df.columns:
    print(f"Frequency Table for {column}")
    print(df[column].value_counts())
    print()

# making a new dataframe 

frequency_table = df.groupby(['rating', 'topic']).size().reset_index(name='Frequency')


sns.barplot(data=frequency_table, x="topic", y="Frequency", hue="rating")
plt.title("Frequency of Combinations in column1 and column2")
plt.xlabel("Topics")
plt.ylabel("Frequency")
plt.show()