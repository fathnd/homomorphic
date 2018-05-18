#pragma once

#include "Store.hpp"
#include "Utils.hpp"

#include <memory>
#include <thread>
#include <unordered_map>

namespace c10d {

class TCPStoreDaemon {

 public:

  explicit TCPStoreDaemon(int storeListenSocket);
  ~TCPStoreDaemon();

  void join();

 protected:

  void run();
  void stop();

  void query(int socket);

  void setHandler(int socket);
  void addHandler(int socket);
  void getHandler(int socket) const;
  void checkHandler(int socket) const;
  void waitHandler(int socket);

  bool checkKeys(const std::vector<std::string>& keys) const;
  void wakeupWaitingClients(const std::string &key);

  std::thread daemonThread_;
  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  // From key -> the list of sockets waiting on it
  std::unordered_map<std::string, std::vector<int>> waitingSockets_;
  // From socket -> number of keys awaited
  std::unordered_map<int, size_t> keysAwaited_;

  std::vector<int> sockets_;
  int storeListenSocket_;
  std::vector<int> controlPipeFd_ {-1, -1};
};

class TCPStore : public Store {

 public:

  explicit TCPStore(const std::string& masterAddr,
                    PortType masterPort,
                    bool isServer = false);

  virtual ~TCPStore();

  void set(
      const std::string& key,
      const std::vector<uint8_t>& value) override;

  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;

  bool check(const std::vector<std::string>& keys) override;

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

 protected:

  bool isServer_;
  int storeSocket_ = -1;
  int masterListenSocket_ = -1;

  std::string tcpStoreAddr_;
  PortType tcpStorePort_;

  // Only needs to be launched as the server
  std::unique_ptr<TCPStoreDaemon> tcpStoreDaemon_ = nullptr;
};

} // namespace c10d
