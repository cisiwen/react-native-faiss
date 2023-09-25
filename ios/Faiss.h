
#ifdef RCT_NEW_ARCH_ENABLED
#import "RNFaissSpec.h"

@interface Faiss : NSObject <NativeFaissSpec>
#else
#import <React/RCTBridgeModule.h>

@interface Faiss : NSObject <RCTBridgeModule>
#endif

@end
