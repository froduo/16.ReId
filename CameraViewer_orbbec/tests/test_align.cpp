#include <libobsensor/ObSensor.hpp>
#include <cstdio>
int main(){
  const char* names[3]={"HW","SW","OFF"};
  OBAlignMode modes[3]={ALIGN_D2C_HW_MODE,ALIGN_D2C_SW_MODE,ALIGN_DISABLE};
  for(int i=0;i<3;i++){
    try{
      auto pipe=std::make_shared<ob::Pipeline>();
      auto cfg=std::make_shared<ob::Config>();
      cfg->enableStream(OB_SENSOR_COLOR);
      cfg->enableStream(OB_SENSOR_DEPTH);
      try{ cfg->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);}catch(...){}
      try{ cfg->setAlignMode(modes[i]); }catch(ob::Error&e){ printf("[%s] setAlignMode fail: %s\n",names[i],e.getMessage()); continue;}
      try{ pipe->start(cfg);}catch(ob::Error&e){ printf("[%s] start fail: %s\n",names[i],e.getMessage()); continue; }
      int cw=0,ch=0,dw=0,dh=0;
      auto fs=pipe->waitForFrameset(1500);
      if(fs){ auto c=fs->getColorFrame(); auto d=fs->getDepthFrame(); if(c){cw=(int)c->getWidth();ch=(int)c->getHeight();} if(d){dw=(int)d->getWidth();dh=(int)d->getHeight();} }
      printf("[%s] START OK  color=%dx%d depth=%dx%d\n",names[i],cw,ch,dw,dh);
      pipe->stop();
    }catch(ob::Error&e){ printf("[%s] ERR %s: %s\n",names[i],e.getName(),e.getMessage()); }
  }
}
