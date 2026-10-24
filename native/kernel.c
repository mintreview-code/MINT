/* Experimental process-local replacement for one fingerprinted NetMHCpan 4.1
 * executable. No on-disk predictor or model is modified. Reconstructed from
 * installed forward_inp disassembly. SIMD runs across hidden neurons, keeping
 * each neuron's addition order. Never compile with fast-math or FMA contraction.
 * Single-thread only; binary ABI-specific research prototype, not production.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>
#include <immintrin.h>

#define I(p,o) (*(int *)((char *)(p)+(o)))
#define P(p,o,t) (*(t *)((char *)(p)+(o)))
#define G(a) (*(volatile int *)(uintptr_t)(a))
typedef struct {void *net; int ni,nh,no,stride; float *w;} Cached;
static Cached cache[128];
static int nc;
static uint64_t calls;
static int verify;
static uint64_t verified;
typedef void (*Forward)(void*,float**,void*,float*,int,int,int,int,int,int,int,int,int,int,void*);
static Forward original;
static Cached *getcache(void *net) {
 for(int c=0;c<nc;c++) if(cache[c].net==net) return &cache[c];
 if(nc==128) abort();
 Cached *c=&cache[nc++]; int *n=P(net,0x910,int*);
 c->net=net;c->ni=n[0];c->nh=n[1];c->no=n[2];c->stride=(n[1]+7)&~7;
 if(I(net,0x908)!=3 || c->ni>4096 || c->nh>1024 || c->no>16) abort();
 c->w=calloc((c->ni+1)*c->stride,sizeof(float)); if(!c->w) abort();
 float ***w=P(net,0x918,float***);
 for(int i=0;i<=c->ni;i++)for(int h=0;h<c->nh;h++)c->w[i*c->stride+h]=w[0][h][i];
 return c;
}
static inline void add(float *s,const float *w,int n) {
 for(int j=0;j<n;j+=8) _mm256_storeu_ps(s+j,_mm256_add_ps(_mm256_loadu_ps(s+j),_mm256_loadu_ps(w+j)));
}
static inline void muladd(float *s,const float *w,float v,int n) {
 __m256 x=_mm256_set1_ps(v);
 for(int j=0;j<n;j+=8) _mm256_storeu_ps(s+j,_mm256_add_ps(_mm256_loadu_ps(s+j),_mm256_mul_ps(x,_mm256_loadu_ps(w+j))));
}
static void fast_forward(void *pep,float **act,void *net,float *inp,int alpha,int start,
 int mhclen,int contextlen,int gaplen,int gappos,int inslen,int inspos,int outidx,int extralen,void *mhc) {
 (void)outidx; calls++;
 Cached *c=getcache(net); int *units=P(net,0x910,int*);
 for(int l=0;l<3;l++)act[l][units[l]]=1.0f;
 memcpy(act[0],inp,c->ni*sizeof(float));
 float sum[1024]={0}; int k=0;
 int *aa=P(pep,0x18,int*), plen=I(pep,0x10), core=G(0x625bf0);
 #define ADD(i) do{int idx=(i);if(idx<0||idx>c->ni)abort();add(sum,c->w+idx*c->stride,c->stride);}while(0)
 #define DENSE(n) do{int nn=(n);for(int d=0;d<nn;d++,k++)muladd(sum,c->w+k*c->stride,inp[k],c->stride);}while(0)
 #define AA(pos) do{int pp=(pos);if(pp>=0&&pp<plen&&aa[pp]>=0)ADD(k+aa[pp]);k+=alpha;}while(0)
 if(G(0x625ba0)>0)DENSE(alpha);
 if(gaplen>0){for(int j=0;j<core;j++)AA(start+j+(j>=gappos?gaplen:0));}
 else if(inslen>0){for(int j=0;j<core;j++){if(j>=inspos&&j<inspos+inslen){k+=alpha;}else{AA(start+j-(j>=inspos+inslen?inslen:0));}}}
 else {for(int j=0;j<core;j++)AA(start+j);}
 if(G(0x625ba0)>0)DENSE(alpha);
 if(G(0x625a40)>=0)DENSE(4);
 if(extralen>0)DENSE(extralen);
 if(G(0x6258e0)>0)DENSE(2);
 if(G(0x625b90)&&G(0x625a10)>0)DENSE(2);
 if(G(0x625a50)&&G(0x625ae0)>0)DENSE(2);
 if(G(0x625f40))DENSE(2);
 int *ctx=P(pep,0x28,int*);
 for(int j=0;j<contextlen;j++,k+=alpha)if(ctx[j]>=0)ADD(k+ctx[j]);
 if(mhclen>0){int *m=P(mhc,0x58,int*);for(int j=0;j<mhclen;j++,k+=alpha)if(m[j]>=0)ADD(k+m[j]);}
 if(k!=c->ni){fprintf(stderr,"feature mismatch %d %d\n",k,c->ni);abort();}
 ADD(c->ni);
 for(int h=0;h<c->nh;h++)act[1][h]=(float)(1.0/(1.0+exp(-(double)sum[h])));
 float ***w=P(net,0x918,float***);
 for(int o=0;o<c->no;o++){float v=0;for(int h=0;h<=c->nh;h++)v+=act[1][h]*w[1][o][h];act[2][o]=(float)(1.0/(1.0+exp(-(double)v)));}
 if(verify){
  float hidden[1024],output[16];memcpy(hidden,act[1],c->nh*sizeof(float));memcpy(output,act[2],c->no*sizeof(float));
  original(pep,act,net,inp,alpha,start,mhclen,contextlen,gaplen,gappos,inslen,inspos,outidx,extralen,mhc);
  if(memcmp(hidden,act[1],c->nh*sizeof(float))||memcmp(output,act[2],c->no*sizeof(float))){fprintf(stderr,"BITWISE_PARITY_FAILURE call=%lu\n",(unsigned long)calls);abort();}
  verified++;
 }
}
__attribute__((constructor)) static void install(void) {
 char exe[4096];ssize_t len=readlink("/proc/self/exe",exe,sizeof(exe)-1);if(len<0)return;exe[len]=0;
 if(!strstr(exe,"nnalign_gaps_pan_play_MA_wgt_MN_v2_two_outputs_context_allelelist_pboth"))return;
 /* Harness checks full executable SHA256. Independently check function prefix. */
 const unsigned char orig[]={0x41,0x57,0x48,0x89,0xd0,0x41,0x56,0x41,0x55,0x41,0x54,0x55,0x44,0x89,0xc5};
 unsigned char *target=(void*)0x4025b0;
 if(memcmp(target,orig,sizeof(orig))){fputs("ABI mismatch\n",stderr);_exit(120);}
 verify=getenv("NATIVE_SIMD_VERIFY")!=NULL;
 if(verify){
  unsigned char *tr=mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(tr==MAP_FAILED)_exit(123);
  memcpy(tr,orig,sizeof(orig));uintptr_t ret=(uintptr_t)target+sizeof(orig);
  /* RAX is live after displaced instructions; use indirect RIP jump instead. */
  tr[15]=0xff;tr[16]=0x25;memset(tr+17,0,4);memcpy(tr+21,&ret,8);
  if(mprotect(tr,4096,PROT_READ|PROT_EXEC))_exit(124);
  original=(Forward)tr;
 }
 unsigned char jump[12]={0x48,0xb8};uintptr_t addr=(uintptr_t)&fast_forward;memcpy(jump+2,&addr,8);jump[10]=0xff;jump[11]=0xe0;
 long pg=sysconf(_SC_PAGESIZE);void *base=(void*)((uintptr_t)target&~(pg-1));
 if(mprotect(base,pg,PROT_READ|PROT_WRITE|PROT_EXEC)){perror("mprotect");_exit(121);}
 memcpy(target,jump,sizeof(jump));if(mprotect(base,pg,PROT_READ|PROT_EXEC))_exit(122);
}
__attribute__((destructor)) static void summary(void){if(calls){fprintf(stderr,"NATIVE_SIMD calls=%lu networks=%d bitwise_verified=%lu hidden_units=",(unsigned long)calls,nc,(unsigned long)verified);for(int i=0;i<nc;i++)fprintf(stderr,"%s%d",i?",":"",cache[i].nh);fputc('\n',stderr);}}
