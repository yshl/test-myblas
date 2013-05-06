ADD_LIBRARY(myblas SHARED myblas1.c myblas2.c myblas3.c)
SET_TARGET_PROPERTIES(myblas PROPERTIES VERSION ${serial} SOVERSION ${soserial})
