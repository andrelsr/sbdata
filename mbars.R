#Bibliotecas utilizadas
library(tibble)
library(tidyverse)
library(lubridate)
library(stringr)
library(cowplot)
library(arules)
library(arulesViz)

# Importa o conjunto de dados a partir do arquivo CSV
dataset <- read.csv("BreadBasket_DMS.csv") %>%  mutate(Date=as.Date(Date), Time=hms(Time))

#Visão geral dos dados:
glimpse(dataset)
summary(dataset)
dataset$Date[nrow(dataset)] - dataset$Date[1]

#Exporta CSV com o resumo das informações
write.csv(summary(dataset),"summary.csv")



#Gera e exporta CSV um dataframe com agrupado por itens, ordenado por número de vendas
write.csv(ds_by_Items <- dataset 
          %>% group_by(Item) 
          %>% summarise(Soma_de_vendas = n()) 
          %>% arrange(desc(Soma_de_vendas)),"items2.csv")

#15 itens mais vendidos
head(ds_by_Items,15)

#Limpeza removendo os itens NONE
dset_filtered <- dataset[ !( grepl("NONE", dataset$Item)) , ]

#Gráfico dos itens mais vendidos
gmais_vendidos <- dset_filtered %>% 
  group_by(Item) %>% 
  summarise(Itens_vendidos = n()) %>% 
  arrange(desc(Itens_vendidos)) %>%
  slice(1:15) %>% #15 = 15 itens mais vendidos
  ggplot(aes(x=reorder(Item,desc(Itens_vendidos)),y=Itens_vendidos))+ #Ordena o eixo X do item mais vendidos para o 15º mais vendido
  geom_bar(stat="identity")+ 
  labs(x = "Itens", y = "Vendas")+
  coord_cartesian(ylim = c(0, 6000))+
  scale_y_continuous(breaks=c(0,500,1000,3000,5000,6000))+
  theme(axis.text.x = element_text(angle = 90, vjust=0.2, hjust=0.95))+
  theme(axis.title.x = element_text(vjust=0.1))
gmais_vendidos

test <-dset_filtered %>% 
  group_by(Item) %>% 
  summarise(Itens_vendidos = n()) %>% 
  arrange(desc(Itens_vendidos))
write.csv(test,"vendastest.csv")

gvendas_tempo <- dset_filtered %>% 
  group_by(Date) %>% 
  summarise(Itens_vendidos = n()) %>% 
  ggplot(aes(x=Date,y=Itens_vendidos, width=1))+
  geom_bar(stat="identity")+
  coord_cartesian()+
  labs(x="",y = "Vendas diárias")+
  scale_x_date(date_breaks = "1 month", date_labels = "%b/%Y")
gvendas_tempo

gvendas_tempozoom <- dset_filtered %>% 
  group_by(Date) %>% 
  summarise(Itens_vendidos = n()) %>% 
  filter(Date < "2017-01-05" & Date > "2016-12-20"  )%>%
  ggplot(aes(x=Date,y=Itens_vendidos, width=1))+
  geom_bar(stat="identity")+
  labs(x="",y = "Vendas diárias")+
  coord_cartesian()+
  scale_x_date(date_breaks = "2 day", date_labels = "%d/%b")
  theme(axis.text.x = element_text(angle = 90, vjust=0.2, hjust=0.95))+
  theme(axis.title.x = element_text(vjust=0.1))
gvendas_tempozoom

plot_grid(gvendas_tempo, gvendas_tempozoom + theme(axis.title.y = element_blank()) , labels = "AUTO")

gvendas_semanaboxp <- dset_filtered %>%
  group_by(Date) %>% 
  summarise(Itens_vendidos = n()) %>% 
  mutate(DiaSemana = wday(Date, label=T)) %>%
  ggplot(aes(x=DiaSemana,y=Itens_vendidos))+
  geom_boxplot()+
  stat_summary(fun.y=mean, geom="point", shape=4, size=3)+
  labs(x="Dia da Semana",y = "Vendas diárias")+
  theme(legend.position = "right")+
  geom_point(aes(shape = "média"), alpha = 0)+
  guides(shape=guide_legend(title=NULL, override.aes = list(shape=4, alpha = 1)))
gvendas_semanaboxp


#Market Basket Analysis
#Exporta CSV com o dataset limpo
write.table(dset_filtered, file="./dset_filtered.csv",sep=",",quote=FALSE,row.names=FALSE)
#Importa transações como matriz esparsa
TIDs <- read.transactions ("./dset_filtered.csv",format = "single", sep=",", cols = c(3,4))
summary(TIDs)
summary(itemFrequency(TIDs))
#Extração das regras de associação. Suporte mínimo = 25 transações e Confiança mínima = 40%. minlen=2 para evitar LHS vazio.
minsup=25/length(TIDs)
RA <- apriori (TIDs, parameter = list(support=minsup,confidence=0.4,minlen=2)) 
inspect(head(RA,by="support",n=10))

plot(RA, measure=c("support","confidence"), shading="lift", col=grey.colors(100))

plot(RA, method="graph", arrowSize=0.4, alpha=1, main="Grafo para 39 regras")
RA10rules <- head(sort(RA, by="support"),10)
plot(RA10rules, method="graph", arrowSize=0.8, alpha=1, main="Grafo para dez regras com maior \nSuporte \n")
RA10rulesconf <- head(sort(RA, by="confidence"),10)
plot(RA10rulesconf, method="graph", arrowSize=0.8, alpha=1, main="Grafo para dez regras com maior \nConfiança \n")


# library(arules)
# search()
# unloadNamespace("arulesViz")
# unloadNamespace("arules")
# update.packages("arules")
# update.packages("arulesViz")
# 
# library(arulesViz)
