HEADERS += \
    $$PWD/qlog4cplus.h

SOURCES += \
    $$PWD/qlog4cplus.cpp

INCLUDEPATH += $$PWD/include


CONFIG(release,debug|release){
    LIBS += $$PWD/Release/log4cplusU.lib
}

CONFIG(debug, debug|release) {
    LIBS += $$PWD/Debug/log4cplusUD.lib
}

