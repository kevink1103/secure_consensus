<template>
  <div class="home">
    <Banner/>
    <h1>Consensus Algorithm</h1>
    <b-form-group>
      <b-form-radio-group
        id="btn-radios-2"
        v-model="selected"
        :options="options"
        buttons
        button-variant="outline-dark"
        size="mg"
        name="radio-btn-outline"
      ></b-form-radio-group>
    </b-form-group>
    <p v-if="selected=='normal'">
      This is a normal algorithm that no privacy-preserving method is applied.
    </p>
    <p v-if="selected=='noise'">
      This is a privacy-preserving average consensus algorithm that injects noises to initial states.<br>
      <a href="https://yilinmo.github.io/public/papers/tac2014privacy.pdf">Algorithm Reference</a>
    </p>
    <p v-if="selected=='crypto'">
      This is a privacy-preserving average consensus algorithm that facilitates Simple Paillier cryptosystem.<br>
      <a href="https://arxiv.org/pdf/1707.04491.pdf">Algorithm Reference</a>
    </p>

    <br>

    <h1>Algorithm Parameters</h1>
    <transition
        name="fade"
        mode="out-in"
      >
      <Normal v-if="selected=='normal'" ref="normal"/>
      <Noise v-else-if="selected=='noise'" ref="noise"/>
      <Crypto v-else-if="selected=='crypto'" ref="crypto"/>
    </transition>

    <b-button @click="submit" block><h3>Run Simulation</h3></b-button>
  </div>
</template>

<script>
import Banner from '@/components/ui/Banner'
import Normal from '@/components/param/Normal'
import Noise from '@/components/param/Noise'
import Crypto from '@/components/param/Crypto'

export default {
  name: 'Home',
  components: {
    Banner,
    Normal,
    Noise,
    Crypto
  },
  data () {
    return {
      selected: 'normal',
      options: [
        { text: 'Normal', value: 'normal' },
        { text: 'Noise', value: 'noise' },
        { text: 'Crypto', value: 'crypto' }
      ],
    }
  },
  methods: {
    submit() {
      console.log(this.$refs["normal"].$data)
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
input {
  width: 100%;
}
button {
  background-color: #363472;
  height: 65px;
  margin-top: 30px;
  margin-bottom: 40px;
}
button:hover {
  background-color: #23224d;
}
button>h3 {
  margin: 0px;
}
</style>
