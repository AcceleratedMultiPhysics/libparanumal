// Device orchestration for the HCZ binary troubled-element mask.
#ifndef BNS_HCZ_DEVICE_MASK_HPP
#define BNS_HCZ_DEVICE_MASK_HPP
#include "platform.hpp"
#include "ogs.hpp"

namespace libp {

// Match QuadOperator::expand: exchange the old mask before EACH layer, then
// write all owned elements from that snapshot. All ranks call this function
// with the same layers, including ranks without owned elements.
// ids lists every owned element [0,local) exactly once; neighbors use the
// same owned/halo slots as halo. mask includes halos, scratch needs local
// entries and must not alias mask. Initialize OGS Int32 kernels at setup.
// Only owned results are current on return; halos reflect the input to the
// last layer, exactly as on the host. layers=0 performs no work or exchange.
inline void hczExpandDeviceMask(ogs::halo_t& halo, kernel_t& expandKernel,
                                const dlong local,
                                deviceMemory<dlong>& ids,
                                deviceMemory<dlong>& neighbors,
                                deviceMemory<int>& mask,
                                deviceMemory<int>& scratch,
                                const int layers) {
  LIBP_ABORT("HCZ mask buffer layers must be nonnegative",layers<0);
  for(int layer=0;layer<layers;layer++) {
    halo.ExchangeStart(mask,1);
    halo.ExchangeFinish(mask,1);
    if(local) {
      expandKernel(local,ids,neighbors,mask,scratch);
      // Keep the caller's allocation stable and leave exchanged halos alone.
      mask.copyFrom(scratch,local);
    }
  }
}

} // namespace libp
#endif
