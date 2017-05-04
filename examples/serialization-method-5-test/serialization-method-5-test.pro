TEMPLATE = app
CONFIG += console
CONFIG -= qt app_bundle

QMAKE_CXXFLAGS += \
	-std=c++11

INCLUDEPATH += \
	../../include \

SOURCES += \
	main.cpp
