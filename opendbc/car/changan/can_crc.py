"""
can_crc.py – CRC helpers for Changan CAN messages
Reversed from changancan.py call sites:

    can_crc.crc_calculate_crc8(dat[:7])          → CRC-8/SAE-J1850
    can_crc.crc_calculate_crc8(dat[8:15])
    can_crc.crc_calculate_crc8(dat[16:23])
    can_crc.crc_calculate_crc8(dat[24:31])
    can_crc.crc16_ccitt_false(dat[:54])           → CRC-16/CCITT-FALSE (GW_244 ADS check)

Evidence for CRC-8/SAE-J1850:
    - 8-byte CAN frames; CRC stored in byte 7 covering bytes 0-6
    - Polynomial 0x1D (SAE J1850), init=0xFF, no input/output inversion
    - Widely used in automotive gateway messages matching this pattern
    - Same polynomial family used by Toyota/Lexus CAN (likely Veoneer MPC reuse)

Evidence for CRC-16/CCITT-FALSE:
    - Used only for ADS_CRCCheck_244: covers dat[:54] (6.75 CAN frames)
    - 16-bit checksum in GW_244 multi-frame message
    - CCITT-FALSE (poly=0x1021, init=0xFFFF) is standard for AUTOSAR CRC16
"""


class CanCrc:
  _CRC8_TABLE = None

  @classmethod
  def _build_crc8_table(cls) -> list:
    table = []
    for i in range(256):
      crc = i
      for _ in range(8):
        if crc & 0x80:
          crc = ((crc << 1) ^ 0x1D) & 0xFF
        else:
          crc = (crc << 1) & 0xFF
      table.append(crc)
    return table

  def _get_crc8_table(self) -> list:
    if CanCrc._CRC8_TABLE is None:
      CanCrc._CRC8_TABLE = self._build_crc8_table()
    return CanCrc._CRC8_TABLE

  def crc_calculate_crc8(self, data: bytes) -> int:
    """
    CRC-8/SAE-J1850 (poly=0x1D, init=0xFF, no final XOR).

    Used for:
      GW_17E  : ACC_CRCCheck_17E  covers byte[0:7]
      GW_1BA  : ACC_CRCCheck_1BA  covers byte[0:7]
      GW_244  : ACC_CRCCheck_24E  covers byte[0:7]
                ACC_CRCCheck_25E  covers byte[8:15]
      GW_307  : ACC_CRCCheck_35E  covers byte[0:7]
                ACC_CRCCheck_322  covers byte[8:15]
                ACC_CRCCheck_344  covers byte[16:23]
                ACC_CRCCheck_35F  covers byte[24:31]
      GW_31A  : ACC_CRCCheck_36D  covers byte[0:7]
                ACC_CRCCheck_30A  covers byte[8:15]
                ACC_CRCCheck_30D  covers byte[16:23]
                ACC_CRCCheck_367  covers byte[24:31]
    """
    table = self._get_crc8_table()
    crc = 0xFF
    for byte in data:
      crc = table[crc ^ byte]
    return crc

  _CRC16_TABLE = None

  @classmethod
  def _build_crc16_table(cls) -> list:
    table = []
    for i in range(256):
      crc = i << 8
      for _ in range(8):
        if crc & 0x8000:
          crc = ((crc << 1) ^ 0x1021) & 0xFFFF
        else:
          crc = (crc << 1) & 0xFFFF
      table.append(crc)
    return table

  def _get_crc16_table(self) -> list:
    if CanCrc._CRC16_TABLE is None:
      CanCrc._CRC16_TABLE = self._build_crc16_table()
    return CanCrc._CRC16_TABLE

  def crc16_ccitt_false(self, data: bytes) -> int:
    """
    CRC-16/CCITT-FALSE (poly=0x1021, init=0xFFFF, no input/output inversion).

    Used for:
      GW_244 : ADS_CRCCheck_244  covers byte[0:54]
               16-bit integrity check over the entire ADS multi-frame payload
    """
    table = self._get_crc16_table()
    crc = 0xFFFF
    for byte in data:
      crc = ((crc << 8) ^ table[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc
