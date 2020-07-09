#ifndef LITE_WINDOW_H
#define LITE_WINDOW_H

class LiteWindow {
 public:
  LiteWindow() {}
  virtual ~LiteWindow() {}
  int IsValid() { return 0; }
  void MakeCurrent() {}
  void Create(int x = -1, int y = -1, const char* display = NULL) {}
};

#endif
