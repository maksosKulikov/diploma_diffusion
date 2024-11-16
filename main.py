import train_ddpm
import train_pg

#train_ddpm.train(size=64, batch_size=8, link="datasets/shoes")

#train_pg.train_base(size=64, batch_size=8, link="datasets/shoes")

train_pg.train_distill(size=64, batch_size=4, link="datasets/shoes")
