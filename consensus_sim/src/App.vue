<template>
  <div id="app">
    <div class="navbar">
      <b-navbar toggleable="lg" type="light" variant="faded">
        <b-navbar-brand tag="h1" class="mb-0" :to="{ name: 'Home' }">Consensus Sim</b-navbar-brand>
        
        <div class="collapse navbar-collapse justify-content-end">
          <b-navbar-nav>
            <b-nav-item :to="{ name: 'About' }">About</b-nav-item>
          </b-navbar-nav>
        </div>
      </b-navbar>
    </div>
    
    <b-container>
      <transition
        name="fade"
        mode="out-in"
      >
        <router-view/>
      </transition>
    </b-container>
    
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
@import url('https://fonts.googleapis.com/css?family=Noto+Sans&display=swap');

#app {
  font-family: 'Noto Sans', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin: 0;
  padding: 0;
}

html, body {
  height: 100%;
}
a {
  color: #388E3C;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
.navbar {
  width: 100%;
  padding: 0px;
  height: 50px;
  margin-bottom: 20px;
}
.navbar-brand {
  padding-left: 20px;
}

.custom-range::-webkit-slider-thumb {
  background: #2c3e50;
}
.custom-range::-moz-range-thumb {
  background: #2c3e50;
}
.custom-range::-ms-thumb {
  background: #2c3e50;
}
.fade-enter-active,
.fade-leave-active {
  transition-duration: 0.3s;
  transition-property: opacity;
  transition-timing-function: ease;
}
.fade-enter,
.fade-leave-active {
  opacity: 0
}
</style>
