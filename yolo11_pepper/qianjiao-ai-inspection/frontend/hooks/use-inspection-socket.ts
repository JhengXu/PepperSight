"use client";

import { useEffect, useRef, useState } from "react";

import { WS_BASE } from "@/lib/api";
import type { SocketEvent } from "@/types";

export function useInspectionSocket(onEvent: (event: SocketEvent) => void) {
  const [connected, setConnected] = useState(false);
  const callbackRef = useRef(onEvent);
  callbackRef.current = onEvent;

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let pingTimer: ReturnType<typeof setInterval> | null = null;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      socket = new WebSocket(`${WS_BASE}/ws/inspection`);
      socket.onopen = () => {
        setConnected(true);
        pingTimer = setInterval(() => socket?.readyState === WebSocket.OPEN && socket.send("ping"), 20000);
      };
      socket.onmessage = (message) => {
        try {
          callbackRef.current(JSON.parse(message.data) as SocketEvent);
        } catch {
          // Ignore malformed third-party messages without breaking the live channel.
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (pingTimer) clearInterval(pingTimer);
        if (!cancelled) reconnectTimer = setTimeout(connect, 1800);
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (pingTimer) clearInterval(pingTimer);
      socket?.close();
    };
  }, []);

  return connected;
}

