//! TrustMark payload framing (`trustmark/datalayer.py`): schema bits, BCH split and binary payloads.

use std::sync::OnceLock;

use super::bch::{Bch, BchError};

const BCH_POLYNOMIAL: i64 = 137;
/// Decoder output width.
pub const PAYLOAD_LEN: usize = 100;

/// `DataLayer.decode_bitstream(..., "binary")` for one packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedPacket {
    /// The data bits as a `0`/`1` string, truncated to the schema capacity.
    pub payload: String,
    /// The BCH check passed (possibly after corrections).
    pub present: bool,
    /// The schema from the packet's last two bits (0 BCH_SUPER, 1 BCH_5, 2 BCH_4, 3 BCH_3).
    pub schema: i64,
}

/// Payload capacity in bits of each schema (`schemaCapacity`).
pub const fn schema_capacity(schema: i64) -> usize {
    match schema {
        0 => 40,
        1 => 61,
        2 => 68,
        3 => 75,
        _ => 0,
    }
}

/// `(t, data bits)` for each schema; schema 0 decodes with decoder slot 5, i.e. BCH(8).
const fn layout(schema: i64) -> (i64, usize) {
    match schema {
        1 => (5, 61),
        2 => (4, 68),
        3 => (3, 75),
        _ => (8, 40),
    }
}

fn decoder(t: i64) -> Option<&'static Bch> {
    static CODES: OnceLock<[Option<Bch>; 4]> = OnceLock::new();
    let codes = CODES.get_or_init(|| [3, 4, 5, 8].map(|t| Bch::new(t, BCH_POLYNOMIAL)));
    codes.iter().flatten().find(|code| code.t() == t)
}

fn pack(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| chunk.iter().enumerate().fold(0_u8, |byte, (i, &bit)| byte | (u8::from(bit) << (7 - i))))
        .collect()
}

fn bit_string(bytes: &[u8], limit: usize) -> String {
    bytes.iter().flat_map(|b| (0..8).rev().map(move |i| if b >> i & 1 == 1 { '1' } else { '0' })).take(limit).collect()
}

/// Decode one 100-bit packet the way `DataLayer._decode_text(packet, "binary")` does.
pub fn decode_packet(packet: &[bool]) -> Result<DecodedPacket, BchError> {
    if packet.len() != PAYLOAD_LEN {
        return Err(BchError);
    }
    let schema = i64::from(packet[PAYLOAD_LEN - 2]) * 2 + i64::from(packet[PAYLOAD_LEN - 1]);
    let (t, data_bits) = layout(schema);
    let code = decoder(t).ok_or(BchError)?;
    let mut data = pack(&packet[..data_bits]);
    let ecc = pack(&packet[data_bits..96]);
    let capacity = schema_capacity(schema);
    let flips = if ecc.len() as i64 == code.ecc_bytes() { code.decode(&mut data, &ecc)? } else { -1 };
    Ok(DecodedPacket { payload: bit_string(&data, capacity), present: flips != -1, schema })
}

/// `DataLayer(100, encoding_mode=schema).encode_binary([bits])` as 0/1 values.
pub fn encode_packet(bits: &str, schema: i64) -> Option<Vec<bool>> {
    let (t, _) = layout(schema);
    let code = decoder(if schema == 0 { 8 } else { t })?;
    let ecc_bits = code.ecc_bits() as usize;
    let data_bits = PAYLOAD_LEN - ecc_bits - 4;
    let mut data: Vec<bool> = bits.chars().take(data_bits).map(|c| c == '1').collect();
    data.resize(data_bits, false);
    let ecc = code.encode(&pack(&data));
    let mut packet = data;
    packet.extend(ecc.iter().flat_map(|b| (0..8).rev().map(move |i| b >> i & 1 == 1)).take(ecc_bits));
    packet.extend((0..4).rev().map(|i| schema >> i & 1 == 1));
    (packet.len() == PAYLOAD_LEN).then_some(packet)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_schema_round_trips() {
        for schema in 0..4 {
            let capacity = schema_capacity(schema);
            let bits: String = (0..capacity).map(|i| if (i * 7 + 3) % 5 < 2 { '1' } else { '0' }).collect();
            let packet = encode_packet(&bits, schema).expect("encode");
            let decoded = decode_packet(&packet).expect("decode");
            assert_eq!(decoded, DecodedPacket { payload: bits, present: true, schema });
        }
    }
}
