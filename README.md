# Handwriter Identification System


Elyazılarından yazar tanımlama işlemi belgelerin adli analizinde önemli bir konu olmuştur.
Yazar kimliğinin ve yazarın ele geçirilmesindeki zorluklar;

bir kişinin yazı stilini, 

yazarın fiziksel durumunu, 

çoklu görev ve gürültü gibi dikkatleri değiştiren farklı kalemlerin kullanımını 
yazma stilinin yaşla birlikte değişmesini de içerir.
Yazar kimliği, adli tıp alanındaki yazılarda olduğu gibi, yazıların eskilerle karşılaştırılması  gerektiğinde, farklı harfler arasındaki bağlantıların kurulabilmesi için de kullanılabilir. 
Yazar tanıma üzerinde birçok yöntem geliştirilmiş  ve test edilmiştir. 


DeepWriter: Metin Bağımsız Yazar Kimlik Doğrulama için Çok Akışlı Derin CNN


Off-line yazar tanımlaması için yöntemler iki gruba ayrılabilir: metne bağımlı ve metinden bağımsız. metin bağımlı yöntemler;
Metin bağımlı: Tüm yazarların aynı metni yazması gerekmekte.
Metin bağımsız: Hiçbir kısıtlama yok. Eğitim ve test için yazara ait herhangi elyazısı olması yeterli.
Bununla birlikte, metne bağımlı bir metinle karşılaştırıldığında, metinden bağımsız yazar tanımlaması, görüntünün devasa kategoriler arası çeşitlilik sergileyen keyfi metinlerle uğraşması gerekir ve bu nedenle de daha zorlayıcıdır. 
Bu zorlu soruna hitap etmek için, bu yazı, off-line metin bağımsız yazar tanımlaması için etkili sunumları öğrenmek için güçlü bir model olarak Deep CNN (Convolutional Neural Network) kullanmaktadır. 


Deep CNN yapısı, en yeni bilgisayar görüntü problemlerinde, görüntü sınıflandırması, nesne algılama ve  yüz tanıma  dahil olmak üzere büyük bir etki göstermiştir. 

El yazısı tanıma  vb. yazarlar arası tanımlama özelliklerini ayıklamak için çok kanallı bir CNN olan DeepWriter'i öneriyoruz. 

