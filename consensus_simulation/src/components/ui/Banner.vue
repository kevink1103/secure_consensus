<template extends>
  <div class="banner">
    <div>
      <b-jumbotron text-variant="white" border-variant="light">
        <template v-slot:header>Secure and Privacy-Preserving<br>Average Consensus</template>

        <template v-slot:lead>
          Safeguard privacy while reaching consensus in distributed network systems.
        </template>

        <hr class="my-3">

        <p>
          This website will simulate and draw results so that you can observe the performance of each algorithm.
        </p>
      </b-jumbotron>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Banner'
}
</script>

<style scoped>
.jumbotron {
  background-color: #363472;
  padding: 1.5rem 1rem;
  padding-top: 2rem;
}
.jumbotron>h1 {
  font-size: 3rem;
}
@media screen and (max-width: 512px) {
  .jumbotron>h1 {
    font-size: 2.5rem;
  }
}
@media screen and (max-width: 768px) {
  .jumbotron>h1 {
    font-size: 2rem;
  }
}
</style>
