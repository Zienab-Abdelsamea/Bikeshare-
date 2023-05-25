#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate TMDB Movie Dataset
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > **Dataset**: 
# This movie dataset contains information about 10,000 movies including genres, ratings, revenue, budget, and more. It contains movies which are released between 1960 and 2015
# 
# > **Questions**:
# <ul>
#     <li>How are the budget of making a movie changed over time?</li>
#     <li>Which movies have the most and least profit? and Top 15 Movies with highest profit.</li>    
#     <li>What is the most popular keywords?</li> 
#     <li>Is there any relationship between columns?</li>    
#     <li>What is the number of movies released every year?</li>    
#     <li>What is the number of movies for each genre?</li>
#     <li>Which director has the most popular movies?</li>
#     <li> Which production companies have most movies?</li>
# </ul>

# In[19]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > load the data, check for cleanliness, and then trim and clean the dataset for analysis.
# 
# ### General Properties

# In[20]:


#loading the data
df = pd.read_csv("tmdb-movies.csv")
df.head()


# In[21]:


df.info()


# In[22]:


df.describe()


# In[23]:


#checking null values
df.isna().sum()


# In[24]:


#checking duplicated rows
df.duplicated().sum()


# In[25]:


list(df.columns)


# In[ ]:





# ### First thoughts
# * Some columns aren't needed in the analysis which i will drop (imdb_id, budget, revenue, cast, homepage, tagline, overview, vote_count, vote_average, runtime)
# * number of votes differ for each movie , so vote_count and vote_average might not be useful.
# * some rows have NaN values which i will drop.
# * some rows has 0 budget and revanue which i think is a missing data so i will drop them too.
# * there is only 1 duplicate row which i will drop.

# ### Data Cleaning

# In[26]:


#dropping unuseful columns
df.drop(labels =['imdb_id', 'budget', 'revenue', 'cast', 'homepage', 'tagline', 'overview','vote_count', 'vote_average','runtime'], axis=1, inplace=True)
df.shape
df.head()


# In[27]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[28]:


#changing rows with zeros
df['budget_adj'] = df['budget_adj'].replace(0, np.NaN)
df['revenue_adj'] = df['revenue_adj'].replace(0, np.NaN)
#dropping null values
df.dropna(inplace=True)
df.isna().sum()


# In[29]:


#changing the format into DateTime
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_date'].head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Question 1: How are the budget of making a movie changed over time?

# In[30]:


#defining axes
x=df.groupby('release_year').budget_adj.mean().index
y=df.groupby('release_year').budget_adj.mean().values
#size of plot
plt.figure(figsize=(20,10))
plt.bar(x,y);
#x axis label
plt.xlabel('Release Year',size=20)
plt.xticks(size=15)
#y axis label
plt.ylabel('Budget (Adj)',size=20)
plt.yticks(size=15)
#plot label
plt.title('Average Spending over Time',size=15);
sns.set_style("whitegrid")


# #### This plot shows ups and downs many time, we can notice that most spending was in period between 2000 and 2010
# 

# ### Question 2 : Which movies have the most and least profit?

# In[31]:


#making new column for profit
df['profit'] = df['revenue_adj'] - df['budget_adj']
#finding movies with highest and lowest profit 
most =  pd.DataFrame(df.loc[df['profit'].idxmax()])
least = pd.DataFrame(df.loc[df['profit'].idxmin()])
df_profit = pd.concat([most, least],axis=1)
df_profit


# ##### 'Star Wars' movie Directed by George Lucas in 1977 has the most profit,
# ##### 'The Warrior's Way' movie Directed by Sngmoo Lee in 2010 has the least profit (huge loss)

# In[32]:


# creating the dataset
df_mp = pd.DataFrame(df['profit'].sort_values(ascending = False))
df_mp['original_title'] = df['original_title'].copy()

#defining axes
movies = df_mp['original_title'][:10]
profit = df_mp['profit'][:10]

#size of plot  
fig = plt.figure(figsize = (20, 10))

# creating the barh plot
plt.barh(movies, profit, color ='maroon')

#x axis label
plt.xlabel("Profit",size=20)
plt.xticks(size=20)

#y axis label
plt.ylabel("Movies",size=20)
plt.yticks(size=20)

#plot label
plt.title("Top 15 movies with highest profit",size = 20)
sns.set_style("whitegrid")


# ##### from this plot we got most 15 movie with highest profit

# ### Question 3  : What is the most popular keywords

# In[33]:


#using wordcloud
text = df.keywords.str.cat(sep='|')
plt.figure( figsize=(15,10))
wordcloud = WordCloud(width=1280, height=720,background_color="black").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Question 4  : Is there any relationship between columns?

# In[34]:


#correlation heatmap to find relation between columns
corr = df.corr()
plt.figure(figsize=(16, 8))
heatmap = sns.heatmap(corr,annot=True)
heatmap.set_title('correlation between columns', fontdict={'fontsize':15}, pad=12);


# # I will mention just high correlated relationship between columns
# 
# * ***Popularity vs Profit:***
# 
#     It is obvious that we find a relation between them, movies with high popularity has high profit    
#     Correlation = 0.51
#     
#     
# * ***Popularity vs Revenue:***
# 
#     It is obvious that we find a relation between them, movies with high popularity has high revenue    
#     Correlation = 0.54
#     
#     
# * ***Revenue vs Profit:***
# 
#     It is obvious that we find a relation between them, movies with high revenue has high profit    
#     Correlation = 0.98
#     
#     
# * ***Budget vs Revenue:***
# 
#     we can find that there is a acceptable relationship between budget and revenue, movies with higher budget get higher revenue
#     
#     Correlation = 0.57
#     
#     
# 
# * ***Popularity vs Revenue:***
# 
#      we can find that there is a acceptable relationship between budget and revenue, movies with higher popularity get higher revenue 
#      
#      Correlation = 0.54

# ### Question 5  : What is the number of movies released every year?

# In[35]:


#group data by 'release year' and 'id' count
df.groupby('release_year').count()['id'].plot(xticks = np.arange(1960,2016,5))
sns.set(rc={'figure.figsize':(20,10)})
#x label
plt.xlabel('Release year',fontsize = 15)
#y label
plt.ylabel('Number of Movies',fontsize = 15)
#plot label
plt.title("number of movies per year",fontsize = 15)
sns.set_style("whitegrid")


# #### number of movies is increasing every year, dropped after 2011

# ### Question 6  : What is the number of movies for each genre?

# In[36]:


#function for spliting values
def split_values(x):
    #link all the rows of the genres.
    link_rows = df[x].str.cat(sep = '|')
    #split all values
    data = pd.Series(link_rows.split('|'))
    #return each genre and it's count.
    return data.value_counts(ascending=False)


# In[37]:


#call the function we called
genres = split_values('genres')

#plot with kind 'bar'
genres.plot(kind= 'bar',figsize = (20,10),fontsize=15,colormap='Pastel1')
#plot label
plt.title("Number of movies for each genre",fontsize=15)
#x axis label
plt.xlabel('Number Of Movies',fontsize=15)
#y axis label
plt.ylabel("Genres",fontsize= 15)
sns.set_style("whitegrid")


# #### we notice that Drama is the most genre while TV movie is the least.

# ### Question 7  : Which director has the most popular movies?

# In[38]:


#group rows by directior and popularity column
director = df.groupby('director').popularity.sum().sort_values(ascending=False)[:15]
#plot with kind 'barh'
director.plot(kind= 'barh',figsize = (20,10),fontsize=15,colormap='Pastel2')

#plot label
plt.title("best 15 director",fontsize=15)
#x axis label
plt.xlabel('Number of Movies',fontsize=15)
#y axis label
plt.ylabel("Director",fontsize= 15)
sns.set_style("whitegrid")


# #### Christopher Nolan has the most number of movies

# ### Question 8  : Which production companies have most movies?

# In[40]:


#call the function we created beore to split rows and return there counts
production_companies = split_values('production_companies')

#select first 15 value
production_companies.iloc[:15].plot(kind='barh',figsize=(20,10),fontsize=15)
#plot label
plt.title("Number of movies for each company",fontsize=15)
#x axis label
plt.xlabel('Number Of Movies',fontsize=15)
#y axis label
plt.ylabel('Company Name',fontsize=15)
sns.set_style("whitegrid")


# #### From the plot above we notice that (Universal Pictures, Warner Bros) have the most number of movies

# <a id='conclusions'></a>
# ## Conclusions
# 
# #### We can summarize our findings in those few points:
# * Budget of making movies is increasing over time.
# * Star Wars is the most profitable movie ever, The Warrior's Way is the least
# * there are many relations in this dataset:
# <ol>
#     <li> Popularity vs Profit </li>
#     <li> Popularity vs Revenue </li>
#     <li> Revenue vs Profit </li>
#     <li> Budget vs Revenue </li>
#     <li> Popularity vs Revenue </li>
# </ol>
# * Number of movies is increasing every year,but dropped after 2011.
# * Drama is the most genre while TV movie is the least.
# * Most Earning Film Production Companies are:
# <ol>
#     <li> Universal Picture </li>
#     <li> Warner Bros </li>
#     <li> Paramount Pictures </li>
#     <li> 20th Century Studios </li>
# </ol>
# * Directors with the most popular movies:
# <ol>
#     <li> Christopher Nolan </li>
#     <li> Steven Spielberg </li>
#     <li> Peter Jackson </li>
#     <li> Quentin Tarantino </li>    
# </ol>
# 
# ## Limitation:
# 
# * many data that was removed such as rows contained becuase they have null values or zero which refer to null too.
# * It should be noted that these analysis does not imply any causation.

# In[ ]:




