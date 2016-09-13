using UnityEngine;
using System.Collections;
using UnityEngine.UI;

public class BuildingManager : MonoBehaviour {

    // for purchase
    public int cost;
    public string costType;
    public float buildingTime;
    public int buildWorkerCap;
    private string itemInfo; // text of item info to be displayed on button
    private string itemDetail; // item detail next to button

    // for progress; worker assignment to buildings. assign workers upto 10. +2 food per min per worker
    // need switch case to define at the start
    public string buildingType;
    private int assignedWorkers;
    private int assignedAtZero;
    public int progressWorkerCap;
    public string effectType;
    public int effectValue;
    public int progressCoolDownInSec;
    private float timePassed;
    private bool assignable;

    private Transform progressCanvas;
    private Transform progressBar;
    private Image progressBarImg;
    public Transform GUI;

    public StatusManager statusManager;

    public GameObject ETprefab;

    private string effectString;
    public bool isBuilt;
    public bool hasPlaced;
    public Color productionColor;
    public Color buildColor;
    // this is necessary bc start is called little after prefab is instantiated. 
    // this prevents initializing twice.(second time resets to init state)
    private bool initialized;

    public GameObject treeToDestroy;

    // Use this for initialization
    void Start() {
        if (!initialized) {
            Init();
        }
    }

    public void Init() {
        // not placed for building yet.
        hasPlaced = false;
        // not built yet.
        isBuilt = false;
        if (buildingType == "House" || buildingType == "Statue") {
            assignable = false;
        } else {
            assignable = true;
        }
        // need to update values in status after cd effect;
        statusManager = GameObject.Find("Canvas").transform.FindChild("Sigil").FindChild("status").GetComponent<StatusManager>();
        // find the attached progress bar
        progressCanvas = transform.FindChild("ProgressCanvas");
        GUI = progressCanvas.FindChild("GUI");
        progressBar = progressCanvas.FindChild("ProgressBG");
        progressBarImg = progressBar.FindChild("Progress").GetComponent<Image>();
        // set progress bar color to red
        progressBarImg.color = buildColor;
        assignedWorkers = 0;
        assignedAtZero = 0;
        //"Time Left : " + (duration - timePassed) / 60 + " min\n" +
        //itemDetailTemp += "\nWorkers Assigned : ";
        //itemDetail.text = itemDetailTemp + assignedWorkersCount + " / " + maxWorkersCount;
        initialized = true;
    }

    // Update is called once per frame
    void Update() {
        if (assignedWorkers > 0) {
            // if this progress is for building (when isBuilt is false)
            if (!isBuilt) {
                // if build progress complete
                if (timePassed >= buildingTime) {
                    buildComplete();
                    // destory the saved tree on the grid
                    Destroy(treeToDestroy);
                // in progress
                } else {
                    // timePass rate proportionate with numWorkerAssigned
                    timePassed += Time.deltaTime * (1 + (assignedWorkers * 0.1f));
                    progressBarImg.fillAmount = timePassed / buildingTime;
                }
            // if the progress is for production,
            } else {
                // if production progress complete
                if (timePassed >= progressCoolDownInSec) {
                    // reset timer
                    timePassed = 0;
                    statusManager.EffectPlus(effectType, effectValue * assignedAtZero);
                    effectString = effectType + " +" + (effectValue * assignedAtZero).ToString();
                    InitET(effectString);
                // in progress
                } else {
                    // if progress is under 25%, assigning worker affects the effect
                    if (progressBarImg.fillAmount <= 0.25) {
                        assignedAtZero = assignedWorkers;
                    }
                    timePassed += Time.deltaTime;
                    progressBarImg.fillAmount = timePassed / progressCoolDownInSec;
                }
            }
        // no worker assigned deactivate the progress
        } else {
            timePassed = 0;
            progressBar.gameObject.SetActive(false);
        }
    }

    public string GetItemInfo() {
        itemInfo = buildingType + "\nCost: " + cost + " " + costType;
        return itemInfo;
    }

    public string GetItemDetail() {
        string itemDetailTemp = "Duration : " + buildingTime / 60 + " min\n" + effectType + " +" + effectValue;
        if (buildingType == "Farm" || buildingType == "Lumber Mill" || buildingType == "Stone Work") {
            itemDetailTemp += " / worker / " + progressCoolDownInSec + " seconds";
        }
        itemDetail = itemDetailTemp;
        return itemDetail;
    }

    // called when building progress reaches 100 (when building completes).
    public void buildComplete() {
        // destroy trees on hexgrid

        // free all assigned workers that were building
        statusManager.numIdleWorkers += assignedWorkers;
        assignedWorkers = 0;
                        // update workercount text.
                GUI.FindChild("WorkerCount").GetComponent<Text>().text = assignedWorkers.ToString();
        // building is now complete
        isBuilt = true;
        // make the building appear on screen
        if (gameObject.GetComponent<MeshRenderer>()) {
            gameObject.GetComponent<MeshRenderer>().enabled = true;
        } else {
            transform.GetChild(0).gameObject.SetActive(true);
        }
        // reset timer
        timePassed = 0;

        // increase the effect type by its value on status for special buildings
        if (effectType == "Population" || effectType == "Fame") {
            statusManager.EffectPlus(effectType, effectValue);
        }
        if (gameObject.tag == "EffectBuilding") {
            GUI.gameObject.SetActive(false);
        }
        // itemDetail.text = itemDetailTemp + assignedWorkersCount + " / " + buildingManager.buildWorkerCap;
    }

    // right click to remove/sell building
    public void MouseOver() {
        // if building is assignable and complete,
        if (hasPlaced && gameObject.tag == "ProductionBuilding") {
            if (Input.GetMouseButtonDown(0)) {
                if (GUI.gameObject.activeSelf) {
                    GUI.gameObject.SetActive(false);
                } else {
                    GUI.gameObject.SetActive(true);
                }
            }
        } else if (hasPlaced && gameObject.tag == "EffectBuilding") {
            if (Input.GetMouseButtonDown(0)) {
                if (!isBuilt) {
                    if (GUI.gameObject.activeSelf) {
                        GUI.gameObject.SetActive(false);
                    } else {
                        GUI.gameObject.SetActive(true);
                    }
                } else {
                    GUI.gameObject.SetActive(false);
                }
            }
        }
        /*
        // left click, increment
        if (Input.GetMouseButtonDown(0)) {
            AssignWorker();
        }
        if (Input.GetMouseButtonDown(1)) {
            // right click, decrement
            RemoveWorker();
        }
        if (Input.GetMouseButtonDown(2)) {
            // middle click, increment all
            AssignAll();
        }
        */
    }

    public void AssignWorker() {
        if (statusManager.numIdleWorkers > 0) {
            // if building is complete AND ASSIGNABLE (for production type buildings), 
            // clicking assigns worker for PRODUCTION progress.
            if (isBuilt && assignable && assignedWorkers < progressWorkerCap) {
                // activate the bar is not already. This needs to be done at every 1st worker assignment.
                if (!progressBar.gameObject.activeSelf) {
                    progressBar.gameObject.SetActive(true);
                    // progress bar color: green; this is already called when built is complete.
                    progressBarImg.color = productionColor;
                }
                assignedWorkers++;
                // update workercount text.
                GUI.FindChild("WorkerCount").GetComponent<Text>().text = assignedWorkers.ToString();
                statusManager.numIdleWorkers--;
            // if building not begun or is still building, clicking assigns worker for BUILD progress. 
            } else if (!isBuilt && assignedWorkers < buildWorkerCap) {
                // turn off building appearance until build process complete
                if (gameObject.GetComponent<MeshRenderer>()) {
                    if (gameObject.GetComponent<MeshRenderer>().enabled) {
                        gameObject.GetComponent<MeshRenderer>().enabled = false;
                    }
                } else {
                    transform.GetChild(0).gameObject.SetActive(false);
                }
                // activate the bar is not already.
                if (!progressBar.gameObject.activeSelf) {
                    progressBar.gameObject.SetActive(true);
                }
                assignedWorkers++;
                // update workercount text.
                GUI.FindChild("WorkerCount").GetComponent<Text>().text = assignedWorkers.ToString();
                statusManager.numIdleWorkers--;
            }
        }
    }

    public void AssignAll() {
        if (assignedWorkers < progressWorkerCap && statusManager.numIdleWorkers > 0) {
            int numToAssign = progressWorkerCap - assignedWorkers;
            if (statusManager.numIdleWorkers < numToAssign) {
                numToAssign = statusManager.numIdleWorkers;
            }
            statusManager.numIdleWorkers -= numToAssign;
            if (!progressBar.gameObject.activeSelf) {
                progressBar.gameObject.SetActive(true);
            }
            assignedWorkers += numToAssign;
            // update workercount text.
            GUI.FindChild("WorkerCount").GetComponent<Text>().text = assignedWorkers.ToString();
        }
    }

    public void RemoveWorker() {
        // subtract 1 from assigned & assignedAtZero workers if not zero.
        if (assignedWorkers > 0) {
            assignedWorkers--;
            // update workercount text.
            GUI.FindChild("WorkerCount").GetComponent<Text>().text = assignedWorkers.ToString();
            // assigned at zero is 1 <= and <= assignedWorkers
            if (assignedAtZero > 0) {
                assignedAtZero--;
            }
            statusManager.numIdleWorkers++;
            // if in process of being built
            if (!isBuilt && assignedWorkers == 0) {
                // if progress is <= 25%, cancelling returns full amount
                if (progressBarImg.fillAmount <= 0.25f) {
                    statusManager.EffectPlus(costType, cost);
                // else cancel returns 75%
                } else {
                    statusManager.EffectPlus(costType, (int)(cost * 0.75f));
                }
                // destroy the building.
                Destroy(gameObject);
            }
        }
    }

    // effect text pop
    void InitET(string text) {
        GameObject temp = Instantiate(ETprefab) as GameObject;
        RectTransform tempRect = temp.GetComponent<RectTransform>();
        temp.transform.SetParent(transform.FindChild("ProgressCanvas"));
        tempRect.transform.localPosition = ETprefab.transform.localPosition;
        tempRect.transform.localScale = ETprefab.transform.localScale;
        tempRect.transform.localRotation = ETprefab.transform.localRotation;

        temp.GetComponent<Text>().text = text;
        temp.GetComponent<Animator>().SetTrigger("Pop");
        Destroy(temp.gameObject, 2);
    }
}