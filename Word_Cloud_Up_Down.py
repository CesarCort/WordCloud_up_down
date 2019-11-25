# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 06:09:49 2019

@author: César Cortez
"""

import numpy as np
import pandas as pd
from os import path
# Visual LIB
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
# =============================================================================
## READ DATA
# =============================================================================
df = pd.read_csv(cwd+"\\data_csv\Complete_Status_Synlab.csv")
# Si tu colum
df.rename(columns = {"¿Qué feedback le darías a [url('evaluado')]?":"feedback_evaluador",
                    "¿Recomendarías este líder a otros equipos?":"recommend_u_leader"},
                        inplace=True)
# Fillna
df.fillna("",inplace=True)
# More Basic Information
print("Hay  {} observaciones y {} caracterizticas en este dataset. \n".format(df.shape[0],df.shape[1]))
#%%
# =============================================================================
# Basura por acá, no ver
# 
# # # Comparisson by GROUPBY
# # Groupby by country 
# Evaluado = df.groupby("Evaluado")
# # Summary statistic of all countries
# Evaluado.describe().head()
# # This selects the top 5 highest average points among all 44 countries:
# Evaluado.mean().sort_values(by="points",ascending=False).head()
# =============================================================================

# =============================================================================
# WORDCLOUD
# More Information :D
# https://www.datacamp.com/community/tutorials/wordcloud-python
# =============================================================================
# Start with one review:
text = df.feedback_evaluador[0]
text_positive = df_posi.feedback_evaluador[0]
# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# To - Save
wordcloud.to_file("img/word_cloud_white.png")
# =============================================================================
# ## ==>>>> Combine all comments 
# =============================================================================
# separete POSI | NEGA por criterio
# =============================================================================
#%%
text = df.feedback_evaluador[0]
df_posi = df[df["recommend_u_leader"]=="si"]
df_nega = df[df["recommend_u_leader"]=="no"]
text_positive = " ".join(str(review) for review in df_posi.feedback_evaluador)
text_negative = " ".join(str(review) for review in df_nega.feedback_evaluador)
text = " ".join(str(review) for review in df.feedback_evaluador)
print ("Tenemos {} palabras en la combinación de todos los Feedbacks.".format(len(text_positive)))
#%%

# =============================================================================
# COMENZANDO CON EL WORDCLOUD
# =============================================================================
#%%
    # STOPWORDS
        # Cargamos los Stopwords | 
        # Create stopword list:
from nltk.corpus import stopwords
import nltk
# Importante | Descargamos los stopwords -> Spanish
nltk.download('stopwords')
stopwords = set(stopwords.words("spanish"))

stopwords.update(["."," "]) # <<<Podemos agregar más cosas aquí
#%%

# SI NECESITAS HELP XD
dir(WordCloud)
help(WordCloud)


# =============================================================================
# CREATE A WORD CLOUD | IMAGE & COLOR
# =============================================================================
#%%
# Generate a word cloud mask_image
        # NEGATIVE IMAGE | HAND_DOWN
#%%
# Aquí solo crearemos el contorno de la imagen | el color es más adelante ;D
logo_mask = np.array(Image.open("img_input/Hand_down.jpg"))
# Create the WordCloud Object
wordcloud_synlab = WordCloud(stopwords=stopwords, background_color="white", 
                          mode="RGBA", max_words=100,
                          mask=logo_mask,
                          max_font_size=200,
                          font_path="font\Futura Book font.ttf").generate(text_negative)

# create coloring from image
image_colors = ImageColorGenerator(logo_mask)
plt.figure(figsize=[15,15])
# Aquí antes de mostrar el wordcloud imprimimos el color de image_color
plt.imshow(wordcloud_synlab.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
plt.savefig("img_ouput/Hand_down_Synlab_futura.png", format="png")
plt.show()
#%%

            # POSITIVE IMAGE | HAND_UP
#%%
# Crea la carpeta img_input | img_ouput
    # NEGATIVE IMAGE | HAND_DOWN
        # Aquí solo crearemos el contorno de la imagen | el color es más adelante ;D
logo_mask = np.array(Image.open("img_input/Hand_up.jpg"))
# Create the WordCloud Object
wordcloud_synlab = WordCloud(stopwords=stopwords, background_color="white", 
                          mode="RGBA", max_words=100,
                          mask=logo_mask,
                          max_font_size=200,
                          font_path="font\Futura Book font.ttf").generate(text_positive)

# create coloring from image
image_colors = ImageColorGenerator(logo_mask)
plt.figure(figsize=[15,15])
# Aquí antes de mostrar el wordcloud imprimimos el color de image_color
plt.imshow(wordcloud_synlab.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
# store to file
plt.savefig("img_ouput/Hand_up_Synlab_futura.png", format="png")
plt.show()
#%%


