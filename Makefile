LIBMDDIR=libmd
include libmd/Makeheader

ActiveElastocapillarity: ActiveElastocapillarity.cc libmd/libmd.cc libmd/libmd.h libmd/libmd-src/* libmd/libmd-src/md/*
	$(CC) $(CCFLAGS) -o ActiveElastocapillarity ActiveElastocapillarity.cc 

clean: 
	rm ActiveElastocapillarity
