using UnityEngine;
using System.Collections;

public class SpawnShip : MonoBehaviour {

    public GameObject pathShip;
    public GameObject cartSpawner;
    public GameObject shipPrefab;
    private GameObject ship;

    Transform targetPathNode;
    int pathNodeIndex = 0;
    int pathSize;
    float speed = 5f;
    [SerializeField]
    private float timeLeft;
    bool canCall;

    // Use this for initialization
    void Start() {
        pathSize = pathShip.transform.childCount;
        timeLeft = 10f;
        canCall = true;
    }

    // Update is called once per frame
    void Update() {
        if (canCall & ship == null) {
            // spawn ship with time delay
            StartCoroutine(WaitSpawnShip(Random.Range(5, 20)));
        } else if (canCall & ship) {
            // next node is not set.
            if (targetPathNode == null) {
                // either node is not set(beginning or end) or node is reached.
                if (pathNodeIndex < pathSize - 1) {
                    targetPathNode = pathShip.transform.GetChild(pathNodeIndex);
                    pathNodeIndex++;
                // node just before the final/departure node
                } else if (pathNodeIndex == pathSize - 1) {
                    if (timeLeft <= 0) {
                        Depart();
                    } else {
                        if (canCall) {
                            StartCoroutine(WaitCart(2));
                        }
                    }
                // final node
                } else {
                    ReachedGoal();
                }
            // next node is set.
            } else {
                Vector3 dir = targetPathNode.position - ship.transform.localPosition;
                float distThisFrame = speed * Time.deltaTime;
                if (dir.magnitude <= distThisFrame) {
                    // we reached the node
                    targetPathNode = null;
                } else {
                    // Move towards node
                    ship.transform.Translate(dir.normalized * distThisFrame, Space.World);
                    Quaternion targetRotation = Quaternion.LookRotation(dir);
                    ship.transform.rotation = Quaternion.Lerp(ship.transform.rotation, targetRotation, Time.deltaTime * 5);
                }
            }
        }
    }

    void ReachedGoal() {
        Destroy(ship.gameObject);
        pathNodeIndex = 0;
        timeLeft = 10;
    }

    IEnumerator WaitCart(int sec) {
        canCall = false;
        yield return new WaitForSeconds(sec);
        timeLeft -= sec;
        cartSpawner.GetComponent<SpawnCart>().Spawn();
        canCall = true;
    }

    IEnumerator WaitSpawnShip(int sec) {
        canCall = false;
        yield return new WaitForSeconds(sec);
        ship = Instantiate(shipPrefab, transform.position, transform.rotation) as GameObject;
        canCall = true;

    }

    void Depart() {
        targetPathNode = pathShip.transform.GetChild(pathNodeIndex);
        pathNodeIndex++;
    }
}
