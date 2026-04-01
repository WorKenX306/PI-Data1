
EMBB_FEATURES = [
    'Dur', 'TotPkts', 'TotBytes', 'Rate',
    'Load', 'Loss', 'pLoss', 'TcpRtt'
]  # Volume-Handshake Hybrid (12 → 8 selected)

MMTC_FEATURES = [
    'TotPkts', 'Rate', 'SrcGap', 'DstGap',
    'Dur', 'Load', 'Loss', 'TcpRtt'
]  # Timing-Centric Minimal Set

URLLC_FEATURES = [
    'TcpRtt', 'SynAck', 'AckDat', 'Loss',
    'Dur', 'Rate', 'TotPkts', 'TotBytes'
]  # Latency-First Strategy

TON_IOT_FEATURES = [
    'src_bytes',   # volume — DDoS/ransomware outbound spikes
    'dst_bytes',   # volume — C2 commands inbound
    'src_pkts',    # packet count — scanning < 10 pkts, DDoS burst
    'dst_pkts',    # packet count — asymmetric response ratio
    'duration',    # timing — backdoor: persistent long sessions
    'proto',       # protocol — UDP floods, protocol mismatches
    'conn_state',  # TCP state — strongest discriminator (REJ=scan, SF=normal)
    'service',     # application layer — SSH=password, HTTP=injection/XSS
]  # IoT Behavioral Approach

FEATURE_MAP = {
    'eMBB':    EMBB_FEATURES,
    'mMTC':    MMTC_FEATURES,
    'URLLC':   URLLC_FEATURES,
    'TON_IoT': TON_IOT_FEATURES,
}