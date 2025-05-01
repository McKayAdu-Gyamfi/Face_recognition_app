import streamlit as st 
import pickle 
import yaml 
import pandas as pd 

cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
PKL_PATH = cfg['PATH']["PKL_PATH"]
st.set_page_config(layout="wide")

#Load databse 
with open(PKL_PATH, 'rb') as file:
    database = pickle.load(file)